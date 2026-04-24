import { DeepAnalysisView } from "@/components/DeepAnalysisView";

// Deep-mode entry point. Minimal page; the DeepAnalysisView owns
// all state. This route is reachable by typing a URL; it's not yet
// in the TopNav (pending user sign-off on where it belongs — mixing
// it into Chat would dilute the chat flow, but a dedicated tab
// might add noise for users who never use deep mode).
export default function DeepPage() {
  return <DeepAnalysisView />;
}
