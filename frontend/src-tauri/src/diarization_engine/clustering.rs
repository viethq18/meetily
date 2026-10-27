// diarization_engine/clustering.rs
//
// Agglomerative hierarchical clustering (complete linkage) over cosine
// dissimilarities of speaker embeddings, mirroring sherpa-onnx's
// fast-clustering.cc (HCLUST_METHOD_COMPLETE + cutree_cdist with a threshold).
//
// The Lance-Williams update for complete linkage keeps, for each merged pair,
// the maximum distance to every other cluster, which allows an O(n^3) but
// allocation-light implementation (n = number of embeddings, typically a few
// hundred per meeting: ~1 per chunk-second of speech).

/// Cluster normalized embeddings with complete-linkage agglomerative
/// clustering; merges stop once the smallest inter-cluster distance exceeds
/// `threshold` (cosine dissimilarity in [0, 2]).
///
/// Returns cluster labels in [0, num_clusters), ordered by first appearance.
pub fn cluster(embeddings: &[Vec<f32>], threshold: f64) -> Vec<i32> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Condensed pairwise cosine dissimilarity (upper triangle, row-major).
    let mut dist = vec![0f64; (n * (n - 1)) / 2];
    let mut k = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            dist[k] = cosine_dissimilarity(&embeddings[i], &embeddings[j]);
            k += 1;
        }
    }

    // Complete-linkage distance between two active clusters: the maximum
    // pairwise distance over their member points.
    let cluster_dist = |members: &Vec<Vec<usize>>, dist: &Vec<f64>, i: usize, j: usize| -> f64 {
        let mut max = f64::NEG_INFINITY;
        for &a in &members[i] {
            for &b in &members[j] {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let idx = condensed_index(lo, hi, n);
                max = max.max(dist[idx]);
            }
        }
        max
    };

    // Active cluster membership; cluster i starts as the singleton {i}.
    // `members[i]` holds the original point indices of active cluster i.
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<bool> = vec![true; n];
    // Merge until the closest pair is farther than the threshold.
    loop {
        let mut best = f64::INFINITY;
        let mut best_i = usize::MAX;
        let mut best_j = usize::MAX;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let d = cluster_dist(&members, &dist, i, j);
                if d < best {
                    best = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best_i == usize::MAX || best > threshold {
            break;
        }
        // Merge j into i.
        let mut merged = std::mem::take(&mut members[best_j]);
        members[best_i].append(&mut merged);
        active[best_j] = false;
    }

    // Assign labels in order of first appearance.
    let mut labels = vec![0i32; n];
    let mut next_label = 0i32;
    for i in 0..n {
        if !active[i] || members[i].is_empty() {
            continue;
        }
        for &p in &members[i] {
            labels[p] = next_label;
        }
        next_label += 1;
    }
    labels
}

fn condensed_index(i: usize, j: usize, n: usize) -> usize {
    // Index of pair (i < j) in the row-major upper triangle.
    i * n - (i * (i + 1)) / 2 + (j - i - 1)
}

/// Cosine dissimilarity of two vectors (1 - cosine similarity), clamped to 0.
pub fn cosine_dissimilarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0; // treat zero vectors as maximally dissimilar-but-not-2
    }
    let sim = dot / (na.sqrt() * nb.sqrt());
    (1.0 - sim).max(0.0)
}

/// L2-normalize an embedding in place (fast-clustering normalizes rows before
/// computing similarities).
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_tight_groups_split() {
        // Two far-apart groups of identical vectors.
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let embeddings = vec![a.clone(), a.clone(), b.clone(), b.clone(), a.clone()];
        let labels = cluster(&embeddings, 0.5);
        let expected_first = labels[0];
        let expected_second = labels[2];
        assert_ne!(expected_first, expected_second);
        for (idx, l) in labels.iter().enumerate() {
            let want = if idx < 2 || idx == 4 {
                expected_first
            } else {
                expected_second
            };
            assert_eq!(*l, want, "point {idx}");
        }
    }

    #[test]
    fn identical_vectors_stay_together() {
        let v = vec![0.3f32, 0.4, 0.5];
        let labels = cluster(&[v.clone(), v.clone(), v.clone()], 0.5);
        assert!(labels.iter().all(|l| *l == labels[0]));
    }

    #[test]
    fn single_point_gets_label_zero() {
        assert_eq!(cluster(&[vec![1.0f32, 2.0]], 0.5), vec![0]);
    }

    #[test]
    fn labels_are_compact() {
        let embeddings = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.9f32, 0.1],
        ];
        let labels = cluster(&embeddings, 0.3);
        let max = *labels.iter().max().unwrap();
        let distinct: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(max as usize + 1, distinct.len());
    }

    #[test]
    fn condensed_index_matches_direct_scan() {
        let n = 5usize;
        let mut expected = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(condensed_index(i, j, n), expected);
                expected += 1;
            }
        }
    }
}
