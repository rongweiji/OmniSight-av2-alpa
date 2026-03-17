/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy all /api/* requests to the Python backend.
  // This means the browser never needs to know the backend's IP —
  // it just calls /api/... and Next.js forwards it server-side.
  async rewrites() {
    const apiUrl = process.env.API_URL ?? "http://127.0.0.1:8080";
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
