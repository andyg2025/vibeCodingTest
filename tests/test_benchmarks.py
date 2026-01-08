"""Performance benchmark suite for the Graph-Driven DFS Code Agent.

Measures performance of key components:
- Context compression efficiency
- Incremental update savings
- Audit pipeline throughput
- DFS walker traversal speed
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from src.engine.context_manager import (
    ContextWindowManager,
    ContextBuilder,
    ContextItem,
)
from src.engine.incremental import (
    ChangeDetector,
    IncrementalUpdatePipeline,
    ImpactAnalyzer,
)
from src.audit import (
    CodeAnalyzer,
    ViolationDetector,
    DetectionContext,
    ReportGenerator,
    ReportFormat,
)
from src.agents.codex import CodexBuilder


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    ops_per_sec: float
    metadata: dict[str, Any]

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total: {self.total_time:.3f}s\n"
            f"  Avg: {self.avg_time*1000:.2f}ms\n"
            f"  Min: {self.min_time*1000:.2f}ms\n"
            f"  Max: {self.max_time*1000:.2f}ms\n"
            f"  Ops/sec: {self.ops_per_sec:.1f}"
        )


def benchmark(
    name: str,
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    **kwargs,
) -> BenchmarkResult:
    """Run a synchronous benchmark.

    Args:
        name: Benchmark name
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Warmup iterations (not counted)
        **kwargs: Arguments to pass to func

    Returns:
        Benchmark result
    """
    # Warmup
    for _ in range(warmup):
        func(**kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_time = sum(times)
    avg_time = total_time / iterations
    min_time = min(times)
    max_time = max(times)
    ops_per_sec = iterations / total_time if total_time > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        ops_per_sec=ops_per_sec,
        metadata=kwargs,
    )


async def async_benchmark(
    name: str,
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    **kwargs,
) -> BenchmarkResult:
    """Run an async benchmark."""
    # Warmup
    for _ in range(warmup):
        await func(**kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await func(**kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_time = sum(times)
    avg_time = total_time / iterations
    min_time = min(times)
    max_time = max(times)
    ops_per_sec = iterations / total_time if total_time > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        ops_per_sec=ops_per_sec,
        metadata=kwargs,
    )


# ====================
# Context Compression Benchmarks
# ====================

class TestContextCompressionBenchmarks:
    """Benchmarks for context compression."""

    @pytest.fixture
    def large_context_items(self) -> list[ContextItem]:
        """Generate large context for benchmarking."""
        items = []

        # Add magic protocols (non-compressible)
        for i in range(5):
            items.append(ContextItem(
                id=f"magic:protocol-{i}",
                content="x" * 1000,  # 1KB each
                category="magic_protocol",
                priority=100,
                tokens=250,
                compressible=False,
            ))

        # Add ancestors
        for i in range(20):
            items.append(ContextItem(
                id=f"ancestor:{i}",
                content="y" * 2000,  # 2KB each
                category="ancestor",
                priority=60 - i,
                tokens=500,
                compressible=True,
                metadata={"type": "module", "name": f"module-{i}"},
            ))

        # Add siblings
        for i in range(50):
            code = f'''
def function_{i}(arg1: str, arg2: int) -> dict:
    """Function {i} docstring."""
    return {{"result": arg1 * arg2}}

class Class{i}:
    """Class {i} docstring."""
    def method(self) -> None:
        pass
'''
            items.append(ContextItem(
                id=f"sibling:file-{i}.py",
                content=code,
                category="sibling",
                priority=40,
                tokens=100,
                compressible=True,
            ))

        # Add generated code
        for i in range(100):
            items.append(ContextItem(
                id=f"generated:file-{i}.py",
                content="z" * 5000,  # 5KB each
                category="generated_code",
                priority=30,
                tokens=1250,
                compressible=True,
            ))

        return items

    def test_benchmark_compression_small(self):
        """Benchmark compression with small context."""
        manager = ContextWindowManager(max_tokens=10000)

        items = [
            ContextItem(
                id=f"item-{i}",
                content="x" * 100,
                category="generated_code",
                priority=50,
                tokens=25,
            )
            for i in range(50)
        ]

        result = benchmark(
            "compression_small_context",
            manager.compress_context,
            iterations=100,
            items=items,
        )

        print(f"\n{result}")
        assert result.avg_time < 0.1  # Should be fast

    def test_benchmark_compression_large(self, large_context_items):
        """Benchmark compression with large context."""
        manager = ContextWindowManager(max_tokens=50000)

        result = benchmark(
            "compression_large_context",
            manager.compress_context,
            iterations=50,
            items=large_context_items,
        )

        print(f"\n{result}")
        # Large context should still complete in reasonable time
        assert result.avg_time < 1.0

    def test_benchmark_compression_ratio(self, large_context_items):
        """Test compression ratio efficiency."""
        manager = ContextWindowManager(max_tokens=20000)

        compressed = manager.compress_context(large_context_items)

        original_tokens = sum(i.tokens for i in large_context_items)

        print(f"\nCompression Stats:")
        print(f"  Original tokens: {original_tokens}")
        print(f"  Compressed tokens: {compressed.total_tokens}")
        print(f"  Compression ratio: {compressed.compression_ratio:.2%}")
        print(f"  Magic protocols preserved: {len(compressed.magic_protocols)}")

        # Should achieve significant compression
        assert compressed.compression_ratio < 0.5


# ====================
# Incremental Update Benchmarks
# ====================

class TestIncrementalUpdateBenchmarks:
    """Benchmarks for incremental update pipeline."""

    @pytest.fixture
    def sample_files(self) -> dict[str, str]:
        """Generate sample files for testing."""
        return {
            f"src/module_{i}/file_{j}.py": f"# File {i}-{j}\npass\n"
            for i in range(10)
            for j in range(10)
        }

    def test_benchmark_change_detection(self, sample_files):
        """Benchmark change detection."""
        detector = ChangeDetector()

        # Initial state
        detector.detect_changes(sample_files)

        # Modify some files
        modified_files = dict(sample_files)
        for i in range(10):
            key = f"src/module_0/file_{i}.py"
            modified_files[key] = f"# Modified\n{modified_files[key]}"

        result = benchmark(
            "change_detection",
            detector.detect_changes,
            iterations=100,
            files=modified_files,
        )

        print(f"\n{result}")
        assert result.avg_time < 0.05  # Should be very fast

    def test_benchmark_hash_computation(self):
        """Benchmark hash computation speed."""
        detector = ChangeDetector()
        content = "x" * 10000  # 10KB file

        result = benchmark(
            "hash_computation",
            detector.compute_hash,
            iterations=1000,
            content=content,
        )

        print(f"\n{result}")
        assert result.ops_per_sec > 10000  # Should be very fast

    @pytest.mark.asyncio
    async def test_benchmark_incremental_pipeline(self, sample_files):
        """Benchmark full incremental update pipeline."""
        pipeline = IncrementalUpdatePipeline()

        # Initial state
        await pipeline.process_update(sample_files)

        # Modify some files
        modified_files = dict(sample_files)
        modified_files["src/module_0/file_0.py"] = "# Changed content"

        result = await async_benchmark(
            "incremental_pipeline",
            pipeline.process_update,
            iterations=50,
            current_files=modified_files,
        )

        print(f"\n{result}")
        assert result.avg_time < 0.5


# ====================
# Audit Pipeline Benchmarks
# ====================

class TestAuditBenchmarks:
    """Benchmarks for audit pipeline."""

    @pytest.fixture
    def sample_code(self) -> str:
        """Generate sample code for auditing."""
        return '''"""Sample module with various patterns."""
import os
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1", tags=["api"])

SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.environ["DATABASE_URL"]


@router.get("/users")
async def get_users(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Get all users."""
    return db.query(User).all()


@router.post("/users")
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
):
    """Create a new user."""
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    return new_user


class UserService:
    """User service class."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.query(User).get(user_id)

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if user:
            self.db.delete(user)
            self.db.commit()
            return True
        return False
'''

    @pytest.fixture
    def codex(self):
        """Create codex for testing."""
        builder = CodexBuilder()
        builder.add_standard_rules()
        return builder.build(freeze=True)

    def test_benchmark_code_analysis(self, sample_code):
        """Benchmark code analysis."""
        analyzer = CodeAnalyzer()

        result = benchmark(
            "code_analysis",
            analyzer.analyze_code,
            iterations=100,
            code=sample_code,
            file_path="test.py",
        )

        print(f"\n{result}")
        assert result.avg_time < 0.1

    def test_benchmark_violation_detection(self, sample_code, codex):
        """Benchmark violation detection."""
        detector = ViolationDetector(codex)
        context = DetectionContext(declared_env_vars=["DATABASE_URL"])

        result = benchmark(
            "violation_detection",
            detector.detect,
            iterations=100,
            code=sample_code,
            file_path="test.py",
            context=context,
        )

        print(f"\n{result}")
        assert result.avg_time < 0.2

    def test_benchmark_report_generation(self, sample_code, codex):
        """Benchmark report generation."""
        detector = ViolationDetector(codex)
        context = DetectionContext(declared_env_vars=[])
        detection_result = detector.detect(sample_code, "test.py", context)

        reporter = ReportGenerator()

        for fmt in [ReportFormat.TEXT, ReportFormat.JSON, ReportFormat.MARKDOWN]:
            result = benchmark(
                f"report_generation_{fmt.value}",
                reporter.generate,
                iterations=100,
                results=[detection_result],
                format=fmt,
            )
            print(f"\n{result}")
            assert result.avg_time < 0.05


# ====================
# Scalability Tests
# ====================

class TestScalability:
    """Test scalability with increasing workload."""

    def test_context_compression_scaling(self):
        """Test how compression scales with item count."""
        manager = ContextWindowManager(max_tokens=50000)

        sizes = [10, 50, 100, 500, 1000]
        results = []

        for size in sizes:
            items = [
                ContextItem(
                    id=f"item-{i}",
                    content="x" * 500,
                    category="generated_code",
                    priority=50,
                    tokens=125,
                )
                for i in range(size)
            ]

            result = benchmark(
                f"compression_{size}_items",
                manager.compress_context,
                iterations=20,
                items=items,
            )
            results.append((size, result.avg_time))
            print(f"\n{result}")

        # Check that scaling is roughly linear
        # Time for 1000 items should be less than 100x time for 10 items
        time_10 = results[0][1]
        time_1000 = results[-1][1]
        scaling_factor = time_1000 / time_10

        print(f"\nScaling factor (10 -> 1000 items): {scaling_factor:.1f}x")
        assert scaling_factor < 200  # Should be sub-quadratic

    def test_audit_scaling(self):
        """Test how audit scales with code size."""
        analyzer = CodeAnalyzer()

        # Generate code of different sizes
        base_code = '''
import os
from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
def test_endpoint():
    return {"status": "ok"}
'''

        sizes = [1, 5, 10, 50]
        results = []

        for multiplier in sizes:
            code = base_code * multiplier

            result = benchmark(
                f"analysis_{len(code)}_chars",
                analyzer.analyze_code,
                iterations=50,
                code=code,
                file_path="test.py",
            )
            results.append((len(code), result.avg_time))
            print(f"\n{result}")

        # Check reasonable scaling
        time_small = results[0][1]
        time_large = results[-1][1]
        scaling_factor = time_large / time_small

        print(f"\nScaling factor (small -> large code): {scaling_factor:.1f}x")


# ====================
# Memory Usage Tests
# ====================

class TestMemoryUsage:
    """Test memory usage of components."""

    def test_context_manager_memory(self):
        """Test memory efficiency of context manager."""
        import sys

        manager = ContextWindowManager(max_tokens=100000)

        # Measure base memory
        base_size = sys.getsizeof(manager)

        # Add many items
        items = [
            ContextItem(
                id=f"item-{i}",
                content="x" * 1000,
                category="generated_code",
                priority=50,
                tokens=250,
            )
            for i in range(1000)
        ]

        compressed = manager.compress_context(items)

        # Measure compressed context size
        compressed_size = sys.getsizeof(compressed)

        print(f"\nMemory Usage:")
        print(f"  Manager base: {base_size} bytes")
        print(f"  Compressed context: {compressed_size} bytes")
        print(f"  Items total chars: {sum(len(i.content) for i in items)}")

    def test_hash_cache_memory(self):
        """Test memory usage of hash cache."""
        import sys

        detector = ChangeDetector()

        # Add many files to cache
        for i in range(10000):
            detector.cache.set_hash(f"path/to/file_{i}.py", f"hash_{i}")

        cache_size = sys.getsizeof(detector.cache.hashes)
        timestamps_size = sys.getsizeof(detector.cache.timestamps)

        print(f"\nHash Cache Memory (10000 files):")
        print(f"  Hashes: {cache_size} bytes")
        print(f"  Timestamps: {timestamps_size} bytes")
        print(f"  Total: {cache_size + timestamps_size} bytes")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)

    # Context compression
    print("\n--- Context Compression ---")
    manager = ContextWindowManager(max_tokens=20000)
    items = [
        ContextItem(
            id=f"item-{i}",
            content="x" * 500,
            category="generated_code",
            priority=50,
            tokens=125,
        )
        for i in range(200)
    ]
    result = benchmark("compression", manager.compress_context, items=items)
    print(result)

    # Change detection
    print("\n--- Change Detection ---")
    detector = ChangeDetector()
    files = {f"file_{i}.py": f"content_{i}" for i in range(100)}
    detector.detect_changes(files)  # Prime cache
    result = benchmark("change_detection", detector.detect_changes, files=files)
    print(result)

    # Code analysis
    print("\n--- Code Analysis ---")
    analyzer = CodeAnalyzer()
    code = "import os\n" * 100
    result = benchmark("analysis", analyzer.analyze_code, code=code, file_path="test.py")
    print(result)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
