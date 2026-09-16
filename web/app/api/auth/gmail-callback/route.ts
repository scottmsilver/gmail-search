import type {NextRequest} from 'next/server';
import {publicRoute} from '@/lib/publicBoundary';
import {pythonApiUrl} from '@/lib/config';

export const runtime = 'nodejs';
export const revalidate = 0;

async function handleGET(req: NextRequest) {
  if (process.env.GMS_FULL_WORKER_ROUTES !== '1') return new Response('Not found', {status:404});
  const params = new URL(req.url).searchParams;
  if (params.getAll('gmail_consent').length !== 1 || [...params.keys()].some(key => key !== 'gmail_consent')) {
    return new Response('Invalid consent callback', {status:400});
  }
  // The backend verifies the signed handoff against both the invited session
  // and its one-use consent cookie. Never follow or log the handoff URL here.
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/gmail-callback?${params}`, {
    redirect:'manual', cache:'no-store',
    headers:{cookie:req.headers.get('cookie') ?? ''},
  });
  const headers = new Headers({'Cache-Control':'private, no-store','Referrer-Policy':'no-referrer'});
  for (const name of ['location','content-type']) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name,value);
  }
  for (const value of upstream.headers.getSetCookie()) headers.append('set-cookie',value);
  return new Response(upstream.body,{status:upstream.status,headers});
}

export const GET = publicRoute(handleGET);
