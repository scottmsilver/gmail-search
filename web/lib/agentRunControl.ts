export type AgentRun = {runId: string; conversationId: string};

export async function stopAgentRun(run: AgentRun): Promise<void> {
  const response = await fetch('/api/agent/analyze', {
    method: 'DELETE', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run_id: run.runId, conversation_id: run.conversationId}),
  });
  const body = await response.json();
  if (!response.ok || !['completed', 'cancelled', 'failed'].includes(body?.state)) {
    throw new Error('Stopping is not confirmed. Please retry Stop.');
  }
}
