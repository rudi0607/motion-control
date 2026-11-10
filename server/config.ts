import path from "node:path";

export const config = {
  port: Number(process.env.PORT ?? 3000),
  dataDir: path.resolve(process.env.DATA_DIR ?? "./data"),
  maxUploadBytes: Number(process.env.MAX_UPLOAD_MB ?? 200) * 1024 * 1024,
  selfhostedUrl: process.env.SELFHOSTED_WORKER_URL?.trim() || undefined,
  selfhostedToken: process.env.SELFHOSTED_WORKER_TOKEN?.trim() || undefined,
  isProd: process.env.NODE_ENV === "production",
};
