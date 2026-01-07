# Graph-Driven DFS Code Agent Architecture - 项目实施计划书 v1.0

## Executive Summary

本项目旨在构建一个基于**图驱动**与**深度优先搜索（DFS）细化**的新一代 Code Agent 架构，解决传统代码生成系统无法有效处理"隐式依赖"和"超长上下文"的核心痛点。

---

## 1. 项目阶段划分

| 阶段 | 名称 | 核心目标 | 关键交付物 |
|------|------|----------|------------|
| **Phase 0** | 基础设施搭建 | 环境配置、技术选型验证 | 开发环境、CI/CD 流水线、图数据库实例 |
| **Phase 1** | 图数据模型设计 | 定义 Node/Edge Schema、魔法依赖元数据标准 | GraphDB Schema、JSON-LD 元数据规范 |
| **Phase 2** | Agent 核心框架 | 实现 Multi-Agent 协作基座 | Agent 通信协议、状态机、编排引擎 |
| **Phase 3** | DFS 生成引擎 | 实现递归生成与上下文回溯机制 | DFS Walker、Context Aggregator、Backtracking Engine |
| **Phase 4** | 魔法依赖审计系统 | 构建合规校验与冲突检测机制 | Audit Agent、Violation Reporter、Auto-Fixer |
| **Phase 5** | 集成测试与优化 | 端到端验证、性能调优 | 测试套件、性能基准、文档 |

---

## 2. 系统架构设计

### 2.1 Multi-Agent 协作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Orchestration Layer                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Agent Message Bus (Redis/NATS)                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │              │                │                │
         ▼              ▼                ▼                ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
│   Global    │ │  Governance  │ │   Logical    │ │      DFS        │
│  Architect  │ │ & Magic-Link │ │   Designer   │ │ Implementation  │
│   Agent     │ │    Agent     │ │    Agent     │ │     Agent       │
└─────────────┘ └──────────────┘ └──────────────┘ └─────────────────┘
      │                │                │                │
      └────────────────┴────────────────┴────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Graph Database   │
                    │    (Neo4j/Memgraph) │
                    └─────────────────────┘
```

### 2.2 各 Agent 职责定义

#### Global Architect Agent
| 属性 | 描述 |
|------|------|
| **输入** | 用户需求描述、项目约束条件 |
| **职责** | 顶层框架选型、模块边界划分、核心依赖拓扑构建 |
| **输出** | 项目骨架图（Skeleton Graph）、技术栈决策文档 |
| **触发条件** | 项目初始化、重大架构变更 |

#### Governance & Magic-Link Agent (核心创新点)
| 属性 | 描述 |
|------|------|
| **输入** | 骨架图、技术栈决策 |
| **职责** | 识别并定义"魔法依赖"：环境变量注入、中间件拦截、全局钩子、AOP 切面、装饰器模式等 |
| **输出** | 隐式协议节点（Magic Protocol Nodes）、依赖法典（Dependency Codex） |
| **关键能力** | 跨模块副作用分析、隐式契约推断 |

#### Logical Designer Agent
| 属性 | 描述 |
|------|------|
| **输入** | 模块节点、关联的魔法依赖 |
| **职责** | 模块内部文件级拆解、接口定义、类型契约 |
| **输出** | 文件节点图、接口签名、类型定义 |
| **约束** | 必须遵循魔法依赖法典 |

#### DFS Implementation Agent
| 属性 | 描述 |
|------|------|
| **输入** | 当前待实现节点、聚合上下文 |
| **职责** | 执行具体代码生成、递归调用子节点 |
| **输出** | 源代码文件、单元测试 |
| **核心机制** | 上下文回溯、魔法节点横向检索 |

---

## 3. 图数据库建模

### 3.1 Node 类型定义

```cypher
// 核心节点类型
(:Project {id, name, tech_stack[], created_at})
(:Module {id, name, type: "domain|infrastructure|application", layer})
(:File {id, path, language, hash})
(:MagicProtocol {id, name, type: "env|middleware|hook|aop|decorator"})
(:Interface {id, name, signature})
```

### 3.2 Edge 类型定义

```cypher
// 显式依赖 (编译时可见)
(:File)-[:IMPORTS {symbols[]}]->(:File)
(:Module)-[:DEPENDS_ON {strength: "hard|soft"}]->(:Module)

// 隐式魔法依赖 (运行时/框架注入)
(:File)-[:INFLUENCED_BY {
  injection_point: "constructor|method|lifecycle",
  behavior_modification: "input|output|sideEffect",
  activation_condition: "always|conditional",
  priority: 1..100
}]->(:MagicProtocol)

// 实现关系
(:File)-[:IMPLEMENTS]->(:Interface)
(:Module)-[:CONTAINS]->(:File)
```

### 3.3 魔法依赖元数据示例 (JSON-LD)

```json
{
  "@context": "https://vibecoding.dev/schema/magic-dependency/v1",
  "@type": "MagicProtocol",
  "@id": "magic:auth-middleware",
  "name": "AuthenticationMiddleware",
  "protocolType": "middleware",
  "injectionMechanism": {
    "framework": "FastAPI",
    "pattern": "dependency_injection",
    "scope": "request"
  },
  "contract": {
    "provides": {
      "current_user": {
        "type": "User | None",
        "availability": "after_auth_check"
      }
    },
    "requires": {
      "headers": ["Authorization"],
      "envVars": ["JWT_SECRET"]
    },
    "sideEffects": [
      {
        "type": "exception",
        "condition": "invalid_token",
        "raises": "HTTPException(401)"
      }
    ]
  },
  "influencedNodes": [
    "file:src/api/routes/users.py",
    "file:src/api/routes/orders.py"
  ],
  "codex": {
    "rule_id": "AUTH-001",
    "description": "所有受保护路由必须声明对此协议的依赖",
    "enforcementLevel": "strict"
  }
}
```

---

## 4. DFS 递归逻辑与上下文感知

### 4.1 DFS 遍历算法

```python
def dfs_generate(node_id: str, context: AggregatedContext) -> GeneratedCode:
    """
    深度优先递归生成算法
    """
    # Step 1: 获取当前节点及其所有依赖
    node = graph.get_node(node_id)

    # Step 2: 向上回溯 - 收集祖先链上下文
    ancestor_context = collect_ancestor_context(node_id)

    # Step 3: 横向检索 - 获取影响当前节点的所有魔法协议
    magic_protocols = graph.query("""
        MATCH (n:File {id: $node_id})-[:INFLUENCED_BY]->(mp:MagicProtocol)
        RETURN mp
    """, node_id=node_id)

    # Step 4: 聚合上下文
    full_context = context.merge(ancestor_context).inject_magic(magic_protocols)

    # Step 5: 获取子节点依赖顺序 (拓扑排序)
    children = get_ordered_dependencies(node_id)

    # Step 6: 递归生成子节点
    child_results = {}
    for child_id in children:
        child_results[child_id] = dfs_generate(child_id, full_context)

    # Step 7: 生成当前节点代码
    code = llm_generate(node, full_context, child_results)

    # Step 8: 回溯校验 - 检查是否违反魔法协议
    violations = audit_agent.check_compliance(code, magic_protocols)

    if violations:
        # 触发回溯重设计
        return backtrack_and_redesign(node, violations, context)

    return code
```

### 4.2 上下文聚合策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Aggregation Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │   Ancestor   │   │   Sibling    │   │    Magic Protocol    │ │
│  │   Context    │ + │   Context    │ + │      Injections      │ │
│  │  (vertical)  │   │ (horizontal) │   │ (cross-cutting)      │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│         │                  │                     │               │
│         └──────────────────┼─────────────────────┘               │
│                            ▼                                      │
│                 ┌─────────────────────┐                          │
│                 │  Context Compressor │  ← 智能摘要，防止上下文爆炸 │
│                 └─────────────────────┘                          │
│                            │                                      │
│                            ▼                                      │
│                 ┌─────────────────────┐                          │
│                 │   Priority Ranker   │  ← 基于相关性排序          │
│                 └─────────────────────┘                          │
│                            │                                      │
│                            ▼                                      │
│                 ┌─────────────────────┐                          │
│                 │   Final Context     │  ← 输入到 LLM             │
│                 └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 回溯校验 (Backtracking) 机制

| 触发条件 | 处理策略 | 回溯深度 |
|----------|----------|----------|
| 类型签名不匹配 | 重新生成当前节点 | 0 (当前节点) |
| 魔法协议违规 | 向上通知 Magic-Link Agent 调整协议 | 1-2 层 |
| 循环依赖检测 | 触发 Architect Agent 重新设计模块边界 | 全局 |
| 运行时约束冲突 | Logical Designer 调整接口设计 | 1 层 |

---

## 5. 魔法依赖审计机制

### 5.1 Audit Agent 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Audit Agent System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   Codex Loader  │───▶│  Rule Engine    │                 │
│  │  (法典加载器)    │    │  (规则引擎)      │                 │
│  └─────────────────┘    └────────┬────────┘                 │
│                                  │                           │
│  ┌─────────────────┐    ┌────────▼────────┐                 │
│  │  Code Analyzer  │───▶│ Violation       │                 │
│  │  (AST + 语义)    │    │ Detector        │                 │
│  └─────────────────┘    └────────┬────────┘                 │
│                                  │                           │
│  ┌─────────────────┐    ┌────────▼────────┐                 │
│  │  Auto-Fix       │◀───│ Report          │                 │
│  │  Suggester      │    │ Generator       │                 │
│  └─────────────────┘    └─────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 审计规则类型

| 规则类别 | 描述 | 示例 |
|----------|------|------|
| **契约合规** | 检查是否正确使用魔法协议暴露的接口 | 使用 `current_user` 前必须声明 auth 依赖 |
| **副作用一致性** | 确保代码行为与声明的副作用匹配 | 不能在声明无副作用的函数中写数据库 |
| **环境变量审计** | 验证所有环境变量依赖都已声明 | 使用 `os.getenv("KEY")` 必须在魔法节点中声明 |
| **生命周期约束** | 确保资源按正确顺序初始化/销毁 | 数据库连接必须在 app startup 后使用 |

---

## 6. 技术栈建议

### 6.1 核心技术选型

| 组件 | 推荐方案 | 备选方案 | 选型理由 |
|------|----------|----------|----------|
| **Agent 框架** | LangGraph + Claude API | CrewAI, AutoGen | LangGraph 提供最佳的状态机和图编排能力 |
| **图数据库** | Neo4j Community | Memgraph, ArangoDB | 成熟生态、Cypher 查询、Python 驱动完善 |
| **LLM Provider** | Claude claude-sonnet-4-20250514 | GPT-4o, Gemini Pro | 长上下文、代码生成质量、工具调用稳定性 |
| **消息队列** | Redis Streams | NATS, RabbitMQ | 轻量、支持消费者组、持久化 |
| **AST 分析** | tree-sitter | LibCST (Python), ast | 多语言支持、增量解析 |
| **代码验证** | Ruff + MyPy | Black, Pylint | 快速、类型检查严格 |

### 6.2 项目依赖清单

```toml
[project]
name = "vibecoding"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    # Agent Framework
    "langgraph>=0.2.0",
    "langchain-anthropic>=0.3.0",

    # Graph Database
    "neo4j>=5.20.0",

    # API & Async
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "httpx>=0.27.0",

    # Code Analysis
    "tree-sitter>=0.22.0",
    "tree-sitter-python>=0.21.0",

    # Validation & Types
    "pydantic>=2.9.0",
    "ruff>=0.7.0",

    # Utilities
    "structlog>=24.4.0",
    "redis>=5.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "mypy>=1.12.0",
]
```

---

## 7. 关键挑战分析与应对方案

### 7.1 循环依赖处理

**问题描述**：模块 A 依赖 B，B 又依赖 A，导致 DFS 无限递归。

**解决方案**：

```python
class CycleDependencyResolver:
    """
    循环依赖检测与解决器
    """
    def detect_cycles(self, graph) -> List[Cycle]:
        """使用 Tarjan 算法检测强连通分量"""
        return tarjan_scc(graph)

    def resolve_strategy(self, cycle: Cycle) -> Resolution:
        strategies = [
            # 策略 1: 接口抽象
            InterfaceExtraction(),  # 提取公共接口打破循环

            # 策略 2: 依赖倒置
            DependencyInversion(),  # 引入抽象层

            # 策略 3: 事件驱动
            EventDecoupling(),      # 使用事件总线解耦

            # 策略 4: 合并模块
            ModuleMerge(),          # 将强耦合模块合并
        ]
        return self.select_best_strategy(cycle, strategies)
```

**图建模增强**：

```cypher
// 添加"虚拟接口节点"打破循环
CREATE (interface:Interface {name: "SharedContract"})
// 原来: (A)-[:IMPORTS]->(B), (B)-[:IMPORTS]->(A)
// 改为: (A)-[:IMPLEMENTS]->(interface)<-[:DEPENDS_ON]-(B)
```

### 7.2 增量更新机制

**问题描述**：用户修改一个文件后，如何最小化重新生成范围。

**解决方案**：

```
┌─────────────────────────────────────────────────────────────────┐
│                  Incremental Update Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Change Detection                                              │
│     ┌──────────────┐                                             │
│     │ File Watcher │ ──▶ Detect modified files                   │
│     └──────────────┘                                             │
│             │                                                     │
│             ▼                                                     │
│  2. Impact Analysis                                               │
│     ┌──────────────┐                                             │
│     │ Graph Query  │ ──▶ 查询受影响的下游节点                      │
│     │              │     MATCH (changed)-[:IMPORTS*1..3]->(downstream) │
│     └──────────────┘                                             │
│             │                                                     │
│             ▼                                                     │
│  3. Magic Dependency Check                                        │
│     ┌──────────────┐                                             │
│     │ Protocol     │ ──▶ 检查是否破坏魔法协议契约                   │
│     │ Validator    │                                             │
│     └──────────────┘                                             │
│             │                                                     │
│             ▼                                                     │
│  4. Selective Regeneration                                        │
│     ┌──────────────┐                                             │
│     │ DFS Partial  │ ──▶ 仅重新生成受影响子图                      │
│     │ Walker       │                                             │
│     └──────────────┘                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**关键优化**：

| 优化策略 | 描述 | 预期效果 |
|----------|------|----------|
| **Hash-based Caching** | 基于内容 hash 缓存生成结果 | 跳过未变更节点 |
| **Dependency Graph Pruning** | 裁剪不受影响的子图 | 减少 50%+ 重新生成 |
| **Lazy Magic Validation** | 仅在边界节点验证魔法协议 | 降低审计开销 |
| **Parallel Subtree Generation** | 独立子树并行生成 | 利用多核加速 |

### 7.3 上下文窗口管理

**问题描述**：深度 DFS 可能导致上下文累积超过 LLM 窗口限制。

**解决方案**：

```python
class ContextWindowManager:
    MAX_TOKENS = 180_000  # Claude claude-sonnet-4-20250514 上下文限制
    RESERVE_FOR_OUTPUT = 20_000

    def compress_context(self, full_context: Context) -> Context:
        """
        智能上下文压缩策略
        """
        # 1. 保留必须的魔法协议（不可压缩）
        essential = self.extract_magic_protocols(full_context)

        # 2. 对祖先上下文进行摘要
        ancestor_summary = self.summarize_ancestors(
            full_context.ancestors,
            max_tokens=10_000
        )

        # 3. 对兄弟节点只保留接口签名
        sibling_interfaces = self.extract_interfaces_only(
            full_context.siblings
        )

        # 4. 使用 embedding 相似度选择最相关的历史代码
        relevant_code = self.select_by_relevance(
            full_context.generated_code,
            current_node=full_context.current,
            max_items=10
        )

        return CompressedContext(
            magic_protocols=essential,
            ancestor_summary=ancestor_summary,
            sibling_interfaces=sibling_interfaces,
            relevant_code=relevant_code
        )
```

---

## 8. 实施路线图

### Phase 0: 基础设施搭建
- [ ] 初始化项目结构与依赖
- [ ] 配置 Neo4j 开发环境 (Docker)
- [ ] 搭建 Claude API 集成
- [ ] 建立 CI/CD 流水线

### Phase 1: 图数据模型设计
- [ ] 定义完整的 Node/Edge Cypher Schema
- [ ] 实现 JSON-LD 元数据解析器
- [ ] 创建图数据库初始化脚本
- [ ] 编写 Schema 验证测试

### Phase 2: Agent 核心框架
- [ ] 实现 Global Architect Agent
- [ ] 实现 Governance & Magic-Link Agent
- [ ] 实现 Logical Designer Agent
- [ ] 构建 Agent 通信协议 (Redis Streams)
- [ ] 实现编排引擎 (LangGraph StateGraph)

### Phase 3: DFS 生成引擎
- [ ] 实现 DFS Walker 核心算法
- [ ] 构建 Context Aggregator
- [ ] 实现 Backtracking Engine
- [ ] 实现 DFS Implementation Agent
- [ ] 添加循环依赖检测与处理

### Phase 4: 魔法依赖审计系统
- [ ] 实现 Codex Loader (法典加载器)
- [ ] 构建 Rule Engine
- [ ] 实现 Code Analyzer (tree-sitter 集成)
- [ ] 实现 Violation Detector 与 Reporter
- [ ] 构建 Auto-Fix Suggester

### Phase 5: 集成测试与优化
- [ ] 端到端测试 (真实项目生成)
- [ ] 性能基准测试
- [ ] 上下文压缩优化
- [ ] 增量更新测试
- [ ] 文档编写

---

## 9. 确认的设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **目标语言** | Python Only | v1 专注 Python，降低复杂度，快速验证架构 |
| **图数据库部署** | Local Docker | 开发灵活，无网络延迟，免费 |
| **验证项目** | 设计 Demo 项目 | 包含典型魔法依赖的 FastAPI 示例 |

---

## 10. Demo 验证项目设计

### 项目名称: `fastapi-ecommerce-demo`

一个包含典型"魔法依赖"的 FastAPI 电商示例，用于端到端验证系统能力。

### 魔法依赖覆盖场景

| 魔法类型 | 示例场景 | 验证目标 |
|----------|----------|----------|
| **中间件注入** | JWT 认证中间件 → `current_user` | 跨路由隐式依赖传递 |
| **环境变量** | `DATABASE_URL`, `JWT_SECRET` | 运行时配置追踪 |
| **依赖注入** | `Depends(get_db)` 数据库会话 | 生命周期管理 |
| **后台任务** | `BackgroundTasks` 发送邮件 | 异步副作用 |
| **事件钩子** | `@app.on_event("startup")` | 初始化顺序 |
| **ORM 魔法** | SQLAlchemy 关系加载 | 隐式查询生成 |

### 项目结构预览

```
fastapi-ecommerce-demo/
├── src/
│   ├── core/
│   │   ├── config.py       # 环境变量 (Magic: ENV)
│   │   ├── security.py     # JWT 处理 (Magic: MIDDLEWARE)
│   │   └── database.py     # DB 连接 (Magic: LIFECYCLE)
│   ├── api/
│   │   ├── deps.py         # 依赖注入 (Magic: DI)
│   │   └── routes/
│   │       ├── auth.py
│   │       ├── users.py    # 依赖 current_user
│   │       └── orders.py   # 依赖 current_user + db
│   ├── models/
│   │   ├── user.py         # ORM (Magic: RELATIONSHIP)
│   │   └── order.py
│   └── services/
│       └── email.py        # BackgroundTask (Magic: ASYNC)
└── main.py                 # 生命周期钩子
```

---

## 11. 下一步行动

一旦计划获得批准，实施顺序：

1. **创建项目目录结构**
   ```
   src/
   ├── agents/          # Multi-Agent 实现
   ├── graph/           # 图数据库交互
   ├── engine/          # DFS 生成引擎
   ├── audit/           # 审计系统
   └── models/          # Pydantic 数据模型
   ```

2. **启动 Neo4j Docker**
   ```bash
   docker run -d --name neo4j-vibecoding \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:5.20-community
   ```

3. **更新 `pyproject.toml` 添加核心依赖**

4. **实现 Phase 0 PoC**: Global Architect Agent + 基础图建模
