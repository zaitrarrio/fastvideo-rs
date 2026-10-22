// A D1Database stand-in over node:sqlite with the real migrations applied,
// so store tests run Tween's SQL rather than a mock of it.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

type Row = Record<string, unknown>;
type Param = string | number | bigint | null | Uint8Array;

const MIGRATIONS = join(import.meta.dirname, "..", "..", "migrations");

class Statement {
  constructor(
    private readonly db: DatabaseSync,
    readonly sql: string,
    readonly params: Param[] = [],
  ) {}
  bind(...params: unknown[]): Statement {
    return new Statement(
      this.db,
      this.sql,
      params.map((p) => (p === undefined ? null : typeof p === "boolean" ? Number(p) : (p as Param))),
    );
  }
  async first<T = Row>(col?: string): Promise<T | null> {
    const row = this.db.prepare(this.sql).get(...this.params) as Row | undefined;
    if (!row) return null;
    return (col ? row[col] : { ...row }) as T;
  }
  async all<T = Row>(): Promise<{ results: T[]; success: true; meta: object }> {
    const rows = this.db.prepare(this.sql).all(...this.params) as Row[];
    return { results: rows.map((r) => ({ ...r })) as T[], success: true, meta: {} };
  }
  async run(): Promise<{ success: true; meta: { changes: number } }> {
    const r = this.db.prepare(this.sql).run(...this.params);
    return { success: true, meta: { changes: Number(r.changes) } };
  }
  runSync() {
    this.db.prepare(this.sql).run(...this.params);
  }
}

export function testD1(): D1Database {
  const db = new DatabaseSync(":memory:");
  db.exec("PRAGMA foreign_keys = ON;");
  for (const f of readdirSync(MIGRATIONS).filter((f) => f.endsWith(".sql")).sort()) {
    db.exec(readFileSync(join(MIGRATIONS, f), "utf8"));
  }
  const d1 = {
    prepare: (sql: string) => new Statement(db, sql),
    async batch(stmts: Statement[]) {
      db.exec("BEGIN");
      try {
        for (const s of stmts) s.runSync();
        db.exec("COMMIT");
      } catch (e) {
        db.exec("ROLLBACK");
        throw e;
      }
      return stmts.map(() => ({ success: true, meta: {} }));
    },
    async exec(sql: string) {
      db.exec(sql);
      return { count: 0, duration: 0 };
    },
  };
  return d1 as unknown as D1Database;
}
