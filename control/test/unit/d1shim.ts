// A D1Database over node:sqlite for unit tests (the subset fv-control uses).
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";

class Stmt {
  constructor(private db: DatabaseSync, private sql: string, private args: unknown[] = []) {}
  bind(...args: unknown[]) {
    return new Stmt(this.db, this.sql, args);
  }
  private p() {
    return this.db.prepare(this.sql);
  }
  private a() {
    return this.args.map((x) => (x === undefined ? null : typeof x === "boolean" ? (x ? 1 : 0) : x)) as any[];
  }
  async first<T>(col?: string): Promise<T | null> {
    const r = this.p().get(...this.a()) as any;
    if (!r) return null;
    return (col ? r[col] : { ...r }) as T;
  }
  async all<T>() {
    return { results: (this.p().all(...this.a()) as any[]).map((r) => ({ ...r })) as T[], success: true, meta: {} };
  }
  async run() {
    const r = this.p().run(...this.a());
    return { success: true, meta: { changes: Number(r.changes) } };
  }
}
export function d1(): D1Database {
  const db = new DatabaseSync(":memory:");
  db.exec(readFileSync(new URL("../../migrations/0001_init.sql", import.meta.url), "utf8"));
  return {
    prepare: (sql: string) => new Stmt(db, sql),
    batch: async (stmts: Stmt[]) => Promise.all(stmts.map((s) => s.run())),
    exec: async (sql: string) => db.exec(sql),
  } as unknown as D1Database;
}
