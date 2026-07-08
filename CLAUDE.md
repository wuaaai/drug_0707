# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. 开发流程：优先使用 Agent-Skills 插件

本项目已安装 compound-engineering agent-skills 插件套件。开发时优先通过 Skill 工具调用对应的 agent-skill，而非手动执行。

### 常用命令映射

| 场景 | 使用命令 |
|------|---------|
| 代码审查 | `/ce-code-review` 或 `Skill("ce-code-review")` |
| 创建提交 | `/ce-commit` 或 `Skill("ce-commit")` |
| 执行计划 | `/ce-work` 或 `Skill("ce-work")` |
| 制定计划 | `/ce-plan` 或 `Skill("ce-plan")` |
| 沉淀知识 | `/ce-compound` 或 `Skill("ce-compound")` |
| 调试 | `/ce-debug` 或 `Skill("ce-debug")` |
| 测试驱动开发 | `Skill("agent-skills:test-driven-development")` |
| 代码简化 | `Skill("agent-skills:code-simplification")` |
| 安全检查 | `Skill("agent-skills:security-and-hardening")` |
| 增量实现 | `Skill("agent-skills:incremental-implementation")` |
| 计划分解 | `Skill("agent-skills:planning-and-task-breakdown")` |
| 规范驱动开发 | `Skill("agent-skills:spec-driven-development")` |
| 性能优化 | `Skill("agent-skills:performance-optimization")` |
| 多 Agent 并行探索 | `Agent(subagent_type="Explore", ...)` |

### 原则
- 复杂任务先 `/ce-plan` 再 `/ce-work`
- 每次提交前 `/ce-code-review`
- 解决新问题后 `/ce-compound` 沉淀
- 可并行的工作用多个 Agent 同时执行

### 环境
- 使用 `uv` 管理 Python 依赖（`pyproject.toml`, `uv.lock`）
- 运行 Python 使用 `.venv/Scripts/python`（Windows）或 `.venv/bin/python`（Linux）
- 清华镜像源：`pypi.tuna.tsinghua.edu.cn/simple`
