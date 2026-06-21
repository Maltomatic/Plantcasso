import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // serialport ships native bindings; use Node's require instead of bundling it.
  serverExternalPackages: ["serialport"],
};

export default nextConfig;
