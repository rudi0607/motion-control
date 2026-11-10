import path from "node:path";

export const config = {
  port: Number(process.env.PORT ?? 3000),
  dataDir: path.resolve(process.env.DATA_DIR ?? "./data"),
  maxUploadBytes: Number(process.env.MAX_UPLOAD_MB ?? 200) * 1024 * 1024,
  falKey: process.env.FAL_KEY?.trim() || undefined,
  replicateToken: process.env.REPLICATE_API_TOKEN?.trim() || undefined,
  isProd: process.env.NODE_ENV === "production",
};
