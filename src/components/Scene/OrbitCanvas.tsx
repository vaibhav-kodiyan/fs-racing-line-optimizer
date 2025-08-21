import React, { useEffect } from "react";

interface TLE {
  line1?: string;
  line2?: string;
}

interface OrbitCanvasProps {
  tle?: TLE;
}

function propagateOrbit(line1: string, line2: string) {
  // Placeholder for orbit propagation logic
  console.log("Propagating orbit with", line1, line2);
}

function fallbackTrajectory() {
  // Placeholder for fallback trajectory logic
  console.log("Using fallback trajectory");
}

const OrbitCanvas: React.FC<OrbitCanvasProps> = ({ tle }) => {
  useEffect(() => {
    if (tle?.line1 && tle?.line2) {
      propagateOrbit(tle.line1, tle.line2);
    } else {
      // Skip propagation or provide fallback trajectory
      console.warn("TLE is missing line1 or line2; skipping propagation.");
      fallbackTrajectory();
    }
  }, [tle]);

  return <div id="orbit-canvas" />;
};

export default OrbitCanvas;
