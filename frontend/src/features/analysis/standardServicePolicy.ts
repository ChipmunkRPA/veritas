/** Mirrored by engine/service_policy.py and checked by the shared assessment fixtures. */
export const STANDARD_SERVICE_POLICY = {
  version: 'standard-saas-evidence/v2',
  service_tier: 'standard_saas',
  title: 'Preliminary screening - not for court use',
  confidence_notice: 'Findings link observations to their supporting evidence and distinguish AI-leaning, original-leaning and inconclusive results. Displayed percentages are system estimates, not verified accuracy or proof. Unsupported and inconclusive checks remain unresolved.',
  court_notice: 'This automated SaaS report has limited scientific reliability and is not prepared for court filing, proof of authenticity or expert testimony. Do not rely on it as court evidence. Automated checks can miss AI content and flag genuine content.',
  private_engagement_notice: 'A privately researched report, expert opinion or litigation support requires a separate signed engagement with the Chipmunk Verify expert. Scope, methods, fees and deliverables are agreed privately. A private engagement does not guarantee court admissibility.',
  expert_contact: 'sales@chipmunkrpa.com',
  court_use: 'not_prepared_for_court_use',
  separate_expert_engagement_required: true,
} as const

export type StandardServicePolicy = typeof STANDARD_SERVICE_POLICY
