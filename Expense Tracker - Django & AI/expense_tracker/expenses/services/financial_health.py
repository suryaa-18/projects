from collections import defaultdict
from decimal import Decimal

from expenses.models import (
    Expense,
    Budget,
    SavingsGoal
)

class FinancialHealthEngine:


    @staticmethod
    def calculate(user):

        expenses = Expense.objects.filter(
            user=user
        )

        budgets = Budget.objects.filter(
            user=user
        )


        total_spending = sum(
            Decimal(e.amount)
            for e in expenses
        )

        budget_total = sum(
            Decimal(b.monthly_limit)
            for b in budgets
        )


        savings_goals = (
            SavingsGoal.objects.filter(
                user=user
            )
        )

        total_savings = sum(
            goal.current_amount
            for goal in savings_goals
)

        category_spending = defaultdict(Decimal)


        for expense in expenses:

            category_spending[
                expense.category.name
            ] += expense.amount


        notifications = []
        for budget in budgets:

            spent = category_spending.get(
                budget.category.name,
                Decimal(0)
            )

            usage = (
                spent / Decimal(budget.monthly_limit)
            ) * 100


            if usage >= 100:

                notifications.append(
                    f"Budget exceeded for {budget.category.name}"
                )

            elif usage >= 80:

                notifications.append(
                    f"You have used {usage:.0f}% of your {budget.category.name} budget"
                )


        # HEALTH SCORE

        score = 100


        if budget_total > 0:

            utilization = (
                total_spending / budget_total
            ) * 100

            if utilization > 100:
                score -= 35

            elif utilization > 85:
                score -= 20

            elif utilization > 70:
                score -= 10


        if total_savings  > 0:
            score += 10

        score = max(
            min(score, 100),
            0
        )


        if score >= 85:
            health_status = "Excellent"

        elif score >= 70:
            health_status = "Good"

        elif score >= 50:
            health_status = "Average"

        else:
            health_status = "Poor"


        return {

            "health_score": score,

            "health_status": health_status,

            "total_budget": budget_total,

            "total_spending": total_spending,

            "savings": total_savings,

            "notifications": notifications,

            "category_spending": category_spending,
        }