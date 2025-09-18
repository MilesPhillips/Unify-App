import { serve } from "https://deno.land/std@0.177.0/http/server.ts";

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "POST") {
    const formData = await req.json();
    return new Response(`Received: ${JSON.stringify(formData)}`, { status: 200 });
  }
  return new Response("Send a POST request with form data.", { status: 200 });
};

serve(handler, { port: 8000 });
console.log("Server running on http://localhost:8000");
