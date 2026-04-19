import * as React from "react";

import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(({ className, value = 0, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
    role="progressbar"
    aria-valuemin={0}
    aria-valuemax={100}
    aria-valuenow={Math.max(0, Math.min(100, value))}
    {...props}
  >
    <div
      className="h-full bg-primary transition-all"
      style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
    />
  </div>
));
Progress.displayName = "Progress";

export { Progress };
