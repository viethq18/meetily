'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SummaryModelSettings } from '@/components/SummaryModelSettings';
import { TemplateSettings } from '@/components/TemplateSettings';

/**
 * Summary settings container with two sub-tabs:
 * Models (LLM configuration) and Templates (summary template management).
 */
export function SummarySettings() {
  return (
    <div className="mt-6">
      <Tabs defaultValue="models">
        <TabsList className="bg-gray-100 border border-gray-200 rounded-lg p-1 h-auto">
          <TabsTrigger
            value="models"
            className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md px-4 py-1.5 text-sm"
          >
            Models
          </TabsTrigger>
          <TabsTrigger
            value="templates"
            className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md px-4 py-1.5 text-sm"
          >
            Templates
          </TabsTrigger>
        </TabsList>
        <TabsContent value="models" className="mt-4">
          <SummaryModelSettings />
        </TabsContent>
        <TabsContent value="templates" className="mt-4">
          <TemplateSettings />
        </TabsContent>
      </Tabs>
    </div>
  );
}
