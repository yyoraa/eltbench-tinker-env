from pathlib import Path

from eltbench_tinker.local_env import LocalELTEnvironment
from eltbench_tinker.task import ELTTask, ExpectedTable


FIXTURE = Path(__file__).parent / "fixtures" / "customer_revenue"


def make_task() -> ELTTask:
    return ELTTask(
        task_id="customer-revenue",
        instruction=(
            "Load customers.csv and orders.csv and create customer_revenue "
            "with customer_id BIGINT, name VARCHAR, revenue DECIMAL(38,2)."
        ),
        source_dir=FIXTURE,
        expected_tables=(
            ExpectedTable(
                name="customer_revenue",
                columns=(
                    ("customer_id", "BIGINT"),
                    ("name", "VARCHAR"),
                    ("revenue", "DECIMAL(38,2)"),
                ),
                rows=((1, "Ada", 20.00), (2, "Linus", 9.00), (3, "Grace", 0.00)),
                order_by=("customer_id",),
            ),
        ),
    )


def test_credential_free_episode_and_hidden_grader() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        assert "customer_revenue" in env.reset()
        result = env.run(
            "cat > solution.sql <<'SQL'\n"
            "CREATE OR REPLACE TABLE customer_revenue AS\n"
            "SELECT c.customer_id, c.name, "
            "CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) AS revenue\n"
            "FROM read_csv_auto('customers.csv') c\n"
            "LEFT JOIN read_csv_auto('orders.csv') o USING (customer_id)\n"
            "GROUP BY c.customer_id, c.name;\n"
            "SQL\n"
        )
        assert result.exit_code == 0
        env.execute_sql_file("solution.sql")
        grade = env.grade()
        assert grade.reward == 1.0, grade.errors
        assert grade.components == {
            "table_existence": 1.0,
            "schema": 1.0,
            "content": 1.0,
        }
    finally:
        env.cleanup()


def test_partial_reward_does_not_confuse_existence_with_correctness() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        env.reset()
        env.destination.execute(
            "CREATE TABLE customer_revenue(customer_id BIGINT, name VARCHAR, revenue DOUBLE);"
        )
        grade = env.grade()
        assert grade.reward == 0.1
        assert grade.components["table_existence"] == 1.0
        assert grade.components["schema"] == 0.0
        assert grade.components["content"] == 0.0
    finally:
        env.cleanup()


def test_reset_removes_previous_warehouse_state() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        env.reset()
        env.destination.execute("CREATE TABLE leaked(value INTEGER);")
        assert env.destination.table_exists("leaked")
        env.reset()
        assert not env.destination.table_exists("leaked")
    finally:
        env.cleanup()

