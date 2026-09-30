import fs from 'node:fs';
import path from 'node:path';
import { defineConfig, loadEnv } from 'vite';

function acousticRoomsPlugin(rootPath) {
  const rooms = new Map();
  let issue = null;
  if (!rootPath) issue = 'Set ACOUSTICROOMS_ROOT in apps/web/.env.local.';
  else {
    const meshRoot = path.join(rootPath, 'room_mesh_obj_format');
    if (!fs.existsSync(meshRoot)) issue = 'The configured dataset has no room_mesh_obj_format folder.';
    else {
      for (const category of fs.readdirSync(meshRoot, { withFileTypes: true }).filter((e) => e.isDirectory())) {
        const folder = path.join(meshRoot, category.name);
        for (const file of fs.readdirSync(folder, { withFileTypes: true }).filter((e) => e.isFile() && e.name.toLowerCase().endsWith('.obj'))) {
          const id = category.name + '/' + file.name.slice(0, -4);
          const fullPath = path.join(folder, file.name);
          rooms.set(id, { id, category: category.name, name: file.name.slice(0, -4), bytes: fs.statSync(fullPath).size, fullPath });
        }
      }
    }
  }
  const sendJson = (res, status, data) => {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify(data));
  };
  const middleware = (req, res, next) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/api/acousticrooms') {
      return sendJson(res, 200, {
        available: !issue,
        issue,
        rooms: [...rooms.values()].map(({ id, category, name, bytes }) => ({ id, category, name, bytes })),
        rirAvailable: false
      });
    }
    if (url.pathname === '/api/acousticrooms/mesh') {
      const room = rooms.get(url.searchParams.get('id'));
      if (!room) return sendJson(res, 404, { error: 'Room not found.' });
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Length', room.bytes);
      res.setHeader('Cache-Control', 'private, max-age=3600');
      fs.createReadStream(room.fullPath).on('error', () => {
        if (!res.headersSent) sendJson(res, 500, { error: 'Mesh read failed.' });
        else res.destroy();
      }).pipe(res);
      return;
    }
    next();
  };
  return {
    name: 'acousticrooms-local-dataset',
    configureServer(server) { server.middlewares.use(middleware); },
    configurePreviewServer(server) { server.middlewares.use(middleware); }
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'ACOUSTICROOMS_');
  return { plugins: [acousticRoomsPlugin(env.ACOUSTICROOMS_ROOT)] };
});
