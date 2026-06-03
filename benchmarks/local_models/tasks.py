#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Verma. All rights reserved.
# Licensed under the Apache License 2.0.

"""Task definitions for local/edge model benchmarks.

20 multi-turn coding tasks across 3 difficulty tiers:
  - Easy (5): Single-function, 1-2 turns
  - Medium (10): Multi-function with iterative refinement, 3-6 turns
  - Hard (5): Complex multi-file tasks with escalating context, 6-10 turns

Each task is designed to provoke context accumulation in small models.
The key insight: small models (3-8B) tend to repeat themselves, get stuck
in loops, and fail to self-correct — exactly where state-harness shines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Task:
    """A benchmark task for local model evaluation."""
    name: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    turns: list[dict]  # List of {"role": "user", "content": "..."}
    validator: Optional[Callable[[str], bool]] = None
    max_turns: int = 20
    tags: list[str] = field(default_factory=list)


def _safe_exec_check(code: str, test_code: str) -> bool:
    """Safely execute code and run a test against it."""
    try:
        namespace = {}
        # Extract code blocks from markdown if present
        if "```python" in code:
            blocks = code.split("```python")
            code_parts = []
            for block in blocks[1:]:
                end = block.find("```")
                if end != -1:
                    code_parts.append(block[:end].strip())
            code = "\n".join(code_parts)
        elif "```" in code:
            blocks = code.split("```")
            if len(blocks) >= 3:
                code = blocks[1].strip()
                if code.startswith("python\n"):
                    code = code[7:]

        exec(code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# EASY TASKS (5) — Single function, 1-2 turns
# These should NOT trigger state-harness. Used to measure false positives.
# ═══════════════════════════════════════════════════════════════

EASY_TASKS = [
    Task(
        name="fibonacci",
        difficulty="easy",
        description="Write a Fibonacci function",
        turns=[
            {"role": "user", "content": "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. Use 0-indexing (fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, fibonacci(10)=55). Return ONLY the function, no explanation."},
        ],
        validator=lambda code: _safe_exec_check(code, "assert fibonacci(0) == 0; assert fibonacci(1) == 1; assert fibonacci(10) == 55"),
        max_turns=5,
        tags=["single-turn", "simple"],
    ),
    Task(
        name="is_palindrome",
        difficulty="easy",
        description="Check if string is palindrome",
        turns=[
            {"role": "user", "content": "Write a Python function called `is_palindrome(s)` that returns True if the string is a palindrome (case-insensitive, ignoring non-alphanumeric characters). Return ONLY the function."},
        ],
        validator=lambda code: _safe_exec_check(code, "assert is_palindrome('racecar'); assert is_palindrome('A man, a plan, a canal: Panama'); assert not is_palindrome('hello')"),
        max_turns=5,
        tags=["single-turn", "simple"],
    ),
    Task(
        name="flatten_list",
        difficulty="easy",
        description="Flatten a nested list",
        turns=[
            {"role": "user", "content": "Write a Python function called `flatten(lst)` that takes a nested list and returns a flat list. E.g., flatten([1, [2, [3, 4], 5], 6]) → [1, 2, 3, 4, 5, 6]. Return ONLY the function."},
        ],
        validator=lambda code: _safe_exec_check(code, "assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]; assert flatten([]) == []; assert flatten([1, 2, 3]) == [1, 2, 3]"),
        max_turns=5,
        tags=["single-turn", "simple"],
    ),
    Task(
        name="word_count",
        difficulty="easy",
        description="Count word frequencies",
        turns=[
            {"role": "user", "content": "Write a Python function called `word_count(text)` that returns a dictionary of word frequencies. Convert to lowercase, split on whitespace. Return ONLY the function."},
        ],
        validator=lambda code: _safe_exec_check(code, "result = word_count('hello world hello'); assert result['hello'] == 2; assert result['world'] == 1"),
        max_turns=5,
        tags=["single-turn", "simple"],
    ),
    Task(
        name="matrix_multiply",
        difficulty="easy",
        description="Multiply two matrices",
        turns=[
            {"role": "user", "content": "Write a Python function called `mat_mul(a, b)` that multiplies two matrices (list of lists) and returns the result. Do not use numpy. Return ONLY the function."},
        ],
        validator=lambda code: _safe_exec_check(code, "assert mat_mul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]"),
        max_turns=5,
        tags=["single-turn", "simple"],
    ),
]


# ═══════════════════════════════════════════════════════════════
# MEDIUM TASKS (10) — Multi-turn with iterative refinement
# These are where small models start to struggle and spiral.
# ═══════════════════════════════════════════════════════════════

MEDIUM_TASKS = [
    Task(
        name="calculator_class",
        difficulty="medium",
        description="Build a calculator class iteratively",
        turns=[
            {"role": "user", "content": "Write a Python class called `Calculator` with methods `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)`. Division by zero should raise ValueError. Return ONLY the class."},
            {"role": "user", "content": "Now add a `history` attribute that records every operation as a tuple (operation, a, b, result). Add a method `get_history()` that returns the list. Include the complete updated class."},
            {"role": "user", "content": "Now add `undo()` method that removes the last operation from history and returns it. If history is empty, raise IndexError. Also add `power(a, b)` and `modulo(a, b)`. Include the COMPLETE class with ALL previous methods."},
            {"role": "user", "content": "There's a bug — `undo()` should also support redoing. Add `redo()` that re-applies the last undone operation. Maintain a separate redo stack. Include the COMPLETE, FINAL class."},
        ],
        validator=lambda code: _safe_exec_check(code, """
c = Calculator()
assert c.add(2, 3) == 5
assert c.subtract(10, 4) == 6
assert len(c.get_history()) == 2
"""),
        max_turns=10,
        tags=["multi-turn", "class", "iterative"],
    ),
    Task(
        name="linked_list",
        difficulty="medium",
        description="Implement a linked list with operations",
        turns=[
            {"role": "user", "content": "Write a Python class `LinkedList` with `Node` inner class. Support `append(val)`, `prepend(val)`, `to_list()`. Return ONLY the classes."},
            {"role": "user", "content": "Add `delete(val)` (delete first occurrence), `find(val)` (return index or -1), and `reverse()` (in-place). Include the COMPLETE updated code."},
            {"role": "user", "content": "Add `insert_at(index, val)`, `delete_at(index)`, and `__len__()`. Handle edge cases (empty list, out of bounds → raise IndexError). Include ALL code."},
            {"role": "user", "content": "Now add `sort()` (merge sort, in-place on the linked list — not by converting to array), and `has_cycle()` using Floyd's algorithm. Include the COMPLETE implementation."},
        ],
        validator=lambda code: _safe_exec_check(code, """
ll = LinkedList()
ll.append(3); ll.append(1); ll.append(2)
assert ll.to_list() == [3, 1, 2]
ll.reverse()
assert ll.to_list() == [2, 1, 3]
"""),
        max_turns=10,
        tags=["multi-turn", "data-structure", "iterative"],
    ),
    Task(
        name="json_parser",
        difficulty="medium",
        description="Build a simple JSON parser",
        turns=[
            {"role": "user", "content": "Write a Python function `parse_json(s)` that parses a JSON string and returns the Python object. Support: strings, numbers (int and float), booleans, null, arrays, objects. Do NOT use the json module. Return ONLY the function."},
            {"role": "user", "content": "The parser doesn't handle escaped characters in strings (\\n, \\t, \\\", \\\\). Fix this and also handle negative numbers and scientific notation (1e5, 1.5e-3). Include the COMPLETE updated code."},
            {"role": "user", "content": "Add error reporting: raise `JsonParseError(message, position)` with the character position where parsing failed. Handle: unexpected token, unterminated string, trailing comma. Include COMPLETE code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
assert parse_json('42') == 42
assert parse_json('"hello"') == "hello"
assert parse_json('[1, 2, 3]') == [1, 2, 3]
assert parse_json('{"a": 1}') == {"a": 1}
assert parse_json('true') == True
assert parse_json('null') == None
"""),
        max_turns=10,
        tags=["multi-turn", "parser", "complex"],
    ),
    Task(
        name="rate_limiter",
        difficulty="medium",
        description="Implement a token bucket rate limiter",
        turns=[
            {"role": "user", "content": "Write a Python class `RateLimiter` implementing a token bucket algorithm. Constructor takes `rate` (tokens per second) and `capacity` (max tokens). Method `allow()` returns True if request is allowed. Use `time.monotonic()`. Return ONLY the class."},
            {"role": "user", "content": "Add `allow_n(n)` for consuming n tokens at once, `wait()` that blocks until a token is available (use time.sleep), and `remaining()` that returns current token count. Include COMPLETE class."},
            {"role": "user", "content": "Add a `SlidingWindowLimiter` class that limits to N requests per window of W seconds. It should be more precise than token bucket for bursty traffic. Include BOTH classes."},
        ],
        validator=lambda code: _safe_exec_check(code, """
import time
rl = RateLimiter(rate=10, capacity=10)
assert rl.allow() == True
"""),
        max_turns=10,
        tags=["multi-turn", "concurrency", "iterative"],
    ),
    Task(
        name="binary_search_tree",
        difficulty="medium",
        description="Implement BST with balancing",
        turns=[
            {"role": "user", "content": "Write a Python class `BST` with methods `insert(val)`, `search(val) -> bool`, `inorder() -> list`. Return ONLY the class."},
            {"role": "user", "content": "Add `delete(val)`, `min()`, `max()`, `height()`. Handle the three deletion cases (leaf, one child, two children). Include COMPLETE class."},
            {"role": "user", "content": "Add `is_valid_bst()` that verifies the BST property, and `level_order() -> list[list]` that returns level-order traversal as a list of lists. Include COMPLETE class."},
            {"role": "user", "content": "Convert this to an AVL tree. Add rotation methods (`_rotate_left`, `_rotate_right`) and rebalance after every insert and delete. The tree must maintain O(log n) height. Include COMPLETE implementation."},
        ],
        validator=lambda code: _safe_exec_check(code, """
tree = BST()
for v in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(v)
assert tree.search(4) == True
assert tree.search(9) == False
assert tree.inorder() == [1, 3, 4, 5, 6, 7, 8]
"""),
        max_turns=12,
        tags=["multi-turn", "data-structure", "complex"],
    ),
    Task(
        name="regex_engine",
        difficulty="medium",
        description="Build a simple regex matcher",
        turns=[
            {"role": "user", "content": "Write a Python function `regex_match(pattern, text)` that returns True if the ENTIRE text matches the pattern. Support: literal chars, `.` (any char), `*` (zero or more of previous). Do NOT use the `re` module. Return ONLY the function."},
            {"role": "user", "content": "Add support for `+` (one or more), `?` (zero or one), and character classes like `[abc]` and `[a-z]`. Include the COMPLETE updated function."},
            {"role": "user", "content": "Add support for `^` (start anchor), `$` (end anchor), and `\\d`, `\\w`, `\\s` character classes. Also add `regex_search(pattern, text)` that finds the pattern anywhere in text. Include ALL code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
assert regex_match('a.c', 'abc') == True
assert regex_match('a*', 'aaa') == True
assert regex_match('a*', '') == True
assert regex_match('ab', 'abc') == False
"""),
        max_turns=10,
        tags=["multi-turn", "parser", "complex"],
    ),
    Task(
        name="event_emitter",
        difficulty="medium",
        description="Build an event system",
        turns=[
            {"role": "user", "content": "Write a Python class `EventEmitter` with methods `on(event, callback)`, `emit(event, *args)`, and `off(event, callback)`. Return ONLY the class."},
            {"role": "user", "content": "Add `once(event, callback)` that auto-removes after first call, `listeners(event) -> list`, and `remove_all_listeners(event=None)`. Include COMPLETE class."},
            {"role": "user", "content": "Add wildcard support: `on('*', callback)` receives ALL events. Add `emit_async(event, *args)` using asyncio. Add `pipe(other_emitter)` that forwards all events. Include COMPLETE class."},
        ],
        validator=lambda code: _safe_exec_check(code, """
ee = EventEmitter()
results = []
ee.on('test', lambda x: results.append(x))
ee.emit('test', 42)
assert results == [42]
"""),
        max_turns=10,
        tags=["multi-turn", "patterns", "iterative"],
    ),
    Task(
        name="lru_cache",
        difficulty="medium",
        description="Implement LRU cache from scratch",
        turns=[
            {"role": "user", "content": "Write a Python class `LRUCache` with a given capacity. Methods: `get(key) -> value or -1`, `put(key, value)`. When capacity is exceeded, evict the least recently used item. Do NOT use `functools.lru_cache` or `OrderedDict`. Return ONLY the class."},
            {"role": "user", "content": "Add `delete(key) -> bool`, `size() -> int`, `clear()`, and `keys() -> list` (in most-recently-used order). Include COMPLETE class."},
            {"role": "user", "content": "Add TTL (time-to-live) support: `put(key, value, ttl=None)` where ttl is seconds. Expired entries should be evicted on access. Add `cleanup()` to remove all expired. Include COMPLETE code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
cache = LRUCache(2)
cache.put(1, 'a')
cache.put(2, 'b')
assert cache.get(1) == 'a'
cache.put(3, 'c')  # evicts key 2
assert cache.get(2) == -1
"""),
        max_turns=10,
        tags=["multi-turn", "data-structure", "iterative"],
    ),
    Task(
        name="state_machine",
        difficulty="medium",
        description="Build a finite state machine",
        turns=[
            {"role": "user", "content": "Write a Python class `StateMachine` that takes a dict of transitions: {state: {event: next_state}}. Methods: `__init__(transitions, initial_state)`, `trigger(event) -> new_state`, `current_state`. Invalid transitions raise `InvalidTransition`. Return ONLY the class."},
            {"role": "user", "content": "Add `on_enter(state, callback)` and `on_exit(state, callback)` hooks. Add `can_trigger(event) -> bool`. Add `history` list of (state, event, new_state) tuples. Include COMPLETE class."},
            {"role": "user", "content": "Add guard conditions: `add_guard(state, event, predicate)` — transition only happens if predicate returns True, else raise `GuardRejected`. Add `reset()` to go back to initial state. Add `to_dot() -> str` that outputs Graphviz DOT format. Include COMPLETE class."},
        ],
        validator=lambda code: _safe_exec_check(code, """
sm = StateMachine({'locked': {'coin': 'unlocked'}, 'unlocked': {'push': 'locked'}}, 'locked')
assert sm.current_state == 'locked'
sm.trigger('coin')
assert sm.current_state == 'unlocked'
"""),
        max_turns=10,
        tags=["multi-turn", "patterns", "iterative"],
    ),
    Task(
        name="mini_orm",
        difficulty="medium",
        description="Build a tiny in-memory ORM",
        turns=[
            {"role": "user", "content": "Write Python classes `Table` and `Row`. `Table(name, columns)` where columns is a list of column names. Methods: `insert(**kwargs) -> Row`, `all() -> list[Row]`, `find(id) -> Row`. Each row gets an auto-incrementing `id`. Return ONLY the classes."},
            {"role": "user", "content": "Add `where(**filters) -> list[Row]` for exact-match filtering, `update(id, **kwargs) -> Row`, `delete(id) -> bool`, and `count() -> int`. Include COMPLETE code."},
            {"role": "user", "content": "Add `order_by(column, desc=False) -> list[Row]`, `limit(n) -> list[Row]`, and make them chainable: `table.where(age=25).order_by('name').limit(5)`. This requires a `Query` class. Include ALL code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
t = Table('users', ['name', 'age'])
t.insert(name='Alice', age=30)
t.insert(name='Bob', age=25)
assert t.count() == 2
assert t.find(1).name == 'Alice'
"""),
        max_turns=10,
        tags=["multi-turn", "patterns", "complex"],
    ),
]


# ═══════════════════════════════════════════════════════════════
# HARD TASKS (5) — Escalating context, high spiral probability
# These are specifically designed to push small models into spirals.
# ═══════════════════════════════════════════════════════════════

HARD_TASKS = [
    Task(
        name="http_framework",
        difficulty="hard",
        description="Build a micro HTTP framework",
        turns=[
            {"role": "user", "content": "Write a Python micro HTTP framework class called `App`. It should support route registration via decorators: `@app.route('/path', methods=['GET'])`. Include a `handle_request(method, path, body=None)` method that dispatches to the right handler. Support path parameters like `/users/<id>`. Return ONLY the code."},
            {"role": "user", "content": "Add middleware support: `@app.middleware` decorator for functions that receive (request, next) and can modify request/response. Add `Request` and `Response` classes. Middleware should execute in order. Include COMPLETE code."},
            {"role": "user", "content": "Add JSON body parsing, query string parsing, and response helpers: `Response.json(data)`, `Response.text(msg)`, `Response.redirect(url)`. Add error handling middleware that catches exceptions and returns 500. Include ALL code."},
            {"role": "user", "content": "Add route groups/blueprints: `bp = Blueprint('/api/v1')` with its own routes and middleware. Add `app.register_blueprint(bp)`. Also add `before_request` and `after_request` hooks. Include the COMPLETE, FINAL framework."},
            {"role": "user", "content": "Add a `TestClient` class: `client = TestClient(app)` with `client.get('/path')`, `client.post('/path', json={})` etc. Write 10 test cases using this client that verify all features. Include ALL code (framework + tests)."},
            {"role": "user", "content": "Review the entire framework. Find and fix all bugs. Add type hints to every function. Add comprehensive docstrings. Include the COMPLETE, FINAL, PRODUCTION-READY code for every class."},
        ],
        validator=lambda code: _safe_exec_check(code, """
app = App()
@app.route('/hello', methods=['GET'])
def hello(request):
    return Response.text('hi')
resp = app.handle_request('GET', '/hello')
assert resp is not None
"""),
        max_turns=15,
        tags=["multi-turn", "framework", "escalating"],
    ),
    Task(
        name="spreadsheet_engine",
        difficulty="hard",
        description="Build a spreadsheet with formulas",
        turns=[
            {"role": "user", "content": "Write a Python class `Spreadsheet` that supports cells addressed as 'A1', 'B2', etc. Methods: `set(cell, value)`, `get(cell) -> value`. Values can be numbers, strings, or formulas starting with '='. Support `=A1+B1`, `=A1*2`. Return ONLY the class."},
            {"role": "user", "content": "Add functions: `=SUM(A1:A5)`, `=AVG(A1:A5)`, `=MIN(A1:A5)`, `=MAX(A1:A5)`. Implement range parsing (A1:A5 → list of cells). Include COMPLETE code."},
            {"role": "user", "content": "Add circular reference detection: if A1='=B1' and B1='=A1', raise `CircularReferenceError`. Implement using topological sort. Also add `=IF(A1>0, B1, C1)`. Include COMPLETE code."},
            {"role": "user", "content": "Add reactive updates: when a cell changes, all dependent cells should automatically recalculate. Build a dependency graph. Add `to_csv() -> str` and `from_csv(text)` methods. Include ALL code."},
            {"role": "user", "content": "Add `undo()` and `redo()` support for all cell changes. Add `=VLOOKUP(value, range, col_index)`. Write 10 test cases covering formulas, circular refs, reactive updates, undo/redo. Include COMPLETE code."},
            {"role": "user", "content": "The spreadsheet has bugs in the formula parser when handling nested functions like `=SUM(A1, MAX(B1:B5))`. Fix all parsing bugs and add `=CONCATENATE(A1, \" \", B1)` for string operations. Review and fix the ENTIRE codebase. Include FINAL, COMPLETE code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
ss = Spreadsheet()
ss.set('A1', 10)
ss.set('A2', 20)
ss.set('A3', '=A1+A2')
assert ss.get('A3') == 30
"""),
        max_turns=15,
        tags=["multi-turn", "complex", "escalating"],
    ),
    Task(
        name="query_builder",
        difficulty="hard",
        description="Build a SQL query builder with validation",
        turns=[
            {"role": "user", "content": "Write a Python SQL query builder. `Query.select('name', 'age').from_table('users').where('age > 18').build()` should return `'SELECT name, age FROM users WHERE age > 18'`. Support SELECT, FROM, WHERE, ORDER BY, LIMIT. Return ONLY the class."},
            {"role": "user", "content": "Add JOIN support: `.join('orders', 'users.id = orders.user_id')`, `.left_join(...)`, `.right_join(...)`. Add GROUP BY and HAVING. Add subqueries: `.where('id IN', Query.select('user_id').from_table('orders'))`. Include COMPLETE code."},
            {"role": "user", "content": "Add parameterized queries for SQL injection prevention: `.where('age > ?', 18)` should return `('SELECT ... WHERE age > ?', [18])`. Add INSERT, UPDATE, DELETE builders. Include ALL code."},
            {"role": "user", "content": "Add schema validation: `Schema('users', {'id': 'int', 'name': 'str', 'age': 'int'})`. Query builder should validate column names against schema and raise `SchemaError` for unknown columns. Include COMPLETE code."},
            {"role": "user", "content": "Add query optimization hints: detect and warn about SELECT * (suggest explicit columns), missing WHERE on DELETE/UPDATE, and JOINs without indexes. Add `.explain() -> str` that shows the query plan as text. Include ALL code with 10 test cases."},
            {"role": "user", "content": "Review the entire query builder. Fix all edge cases: empty WHERE, multiple GROUP BY columns, nested subqueries in JOIN conditions. Add type hints, docstrings, and ensure the builder is fully immutable (each method returns a new Query). Include FINAL code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
q = Query.select('name', 'age').from_table('users').where('age > 18').build()
assert 'SELECT' in q
assert 'users' in q
"""),
        max_turns=15,
        tags=["multi-turn", "complex", "escalating"],
    ),
    Task(
        name="interpreter",
        difficulty="hard",
        description="Build a simple programming language interpreter",
        turns=[
            {"role": "user", "content": "Write a Python interpreter for a simple language. Support: variable assignment (`x = 5`), arithmetic (`x + y * 2`), print (`print x`). Class `Interpreter` with method `run(code) -> list[str]` (list of print outputs). Return ONLY the code."},
            {"role": "user", "content": "Add if/else: `if x > 5 then print x else print 0 end`. Add while loops: `while x > 0 do x = x - 1 end`. Include COMPLETE code."},
            {"role": "user", "content": "Add functions: `def add(a, b) return a + b end`. Support function calls: `result = add(3, 4)`. Handle scope (local vs global variables). Include COMPLETE code."},
            {"role": "user", "content": "Add string support with concatenation, comparison operators (==, !=, <, >, <=, >=), boolean operators (and, or, not), and arrays: `arr = [1, 2, 3]`, `arr[0]`, `len(arr)`. Include ALL code."},
            {"role": "user", "content": "Add proper error messages with line numbers: `Error at line 5: undefined variable 'z'`. Handle: undefined variables, type errors (adding string + number), division by zero, stack overflow (recursive functions). Include COMPLETE code."},
            {"role": "user", "content": "The interpreter breaks on multi-line function bodies and nested if/while blocks. Fix ALL parsing bugs. Add for loops: `for i = 0 to 10 do print i end`. Write 15 test cases. Include FINAL, COMPLETE, DEBUGGED code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
interp = Interpreter()
output = interp.run('x = 5\\nprint x')
assert output == ['5']
"""),
        max_turns=15,
        tags=["multi-turn", "parser", "escalating"],
    ),
    Task(
        name="reactive_system",
        difficulty="hard",
        description="Build a reactive state management system",
        turns=[
            {"role": "user", "content": "Write a Python reactive system. `Signal(value)` holds a value. `Computed(fn)` derives from signals. `Effect(fn)` runs side effects. When a signal changes, all dependents update automatically. Return ONLY the code."},
            {"role": "user", "content": "Add batching: `with batch():` groups multiple signal changes and only triggers effects once. Add `untrack(fn)` that reads signals without creating dependencies. Include COMPLETE code."},
            {"role": "user", "content": "Add `watch(signal, callback)` that calls callback with (new_value, old_value). Add conditional dependencies: if a computed reads signal A in one branch and signal B in another, dependencies should update dynamically. Include ALL code."},
            {"role": "user", "content": "Add error handling: if a computed throws, the error propagates to dependents as an `ErrorSignal`. Add `cleanup()` in effects for resource management. Add `derived_async(async_fn)` for async computations. Include COMPLETE code."},
            {"role": "user", "content": "Add devtools: `trace()` that returns the dependency graph as a dict. Add memory leak detection: warn if a signal has no dependents but hasn't been garbage collected. Write 15 test cases covering all features. Include FINAL code."},
            {"role": "user", "content": "The reactive system has a bug: diamond dependencies cause double-updates (A -> B, A -> C, B -> D, C -> D; changing A triggers D twice). Fix this using topological sort scheduling. Also fix stale closure bugs in effects. Include the COMPLETE, FINAL, BUG-FREE code."},
        ],
        validator=lambda code: _safe_exec_check(code, """
s = Signal(1)
double = Computed(lambda: s.value * 2)
assert double.value == 2
s.value = 5
assert double.value == 10
"""),
        max_turns=15,
        tags=["multi-turn", "reactive", "escalating"],
    ),
]


# ═══════════════════════════════════════════════════════════════
# ALL TASKS
# ═══════════════════════════════════════════════════════════════

ALL_TASKS = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS

# Convenience accessors
TASKS_BY_DIFFICULTY = {
    "easy": EASY_TASKS,
    "medium": MEDIUM_TASKS,
    "hard": HARD_TASKS,
}

TASKS_BY_NAME = {task.name: task for task in ALL_TASKS}
