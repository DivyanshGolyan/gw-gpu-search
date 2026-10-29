"""Pod-side HTTP server on :8000: GET serves /work/out, PUT or POST /<name> stores the body as /work/in/<name>."""
import http.server, os
os.makedirs('/work/in', exist_ok=True)
class H(http.server.SimpleHTTPRequestHandler):
    def _store(self):
        n = int(self.headers['Content-Length']); p = '/work/in/' + os.path.basename(self.path)
        with open(p + '.part', 'wb') as f:
            while n > 0:
                b = self.rfile.read(min(n, 1 << 20)); f.write(b); n -= len(b)
        os.rename(p + '.part', p); self.send_response(200); self.end_headers(); self.wfile.write(b'ok\n')
    do_PUT = do_POST = _store
    def log_message(self, *a): pass
os.chdir('/work/out'); http.server.ThreadingHTTPServer(('', 8000), H).serve_forever()
