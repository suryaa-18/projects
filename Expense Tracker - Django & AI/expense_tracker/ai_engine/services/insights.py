from collections import defaultdict
from decimal import Decimal

from expenses.models import Expense


class ExpenseInsightsEngine:


    @staticmethod
    def generate_insights(user):

        expenses = Expense.objects.filter(
            user=user
        ).order_by('-date')


        if not expenses.exists():

            return {
                "insights": [
                    "No expenses available yet."
                ]
            }


        insights = []


        # TOTAL SPENDING

        total_spending = sum(
            expense.amount
            for expense in expenses
        )


        insights.append(
            f"Your total spending is ₹{total_spending:,.0f}."
        )


        # CATEGORY TOTALS

        category_totals = defaultdict(Decimal)

        for expense in expenses:

            category_totals[
                expense.category.name
            ] += expense.amount


        # MOST SPENT CATEGORY

        top_category = max(
            category_totals,
            key=category_totals.get
        )


        top_amount = category_totals[
            top_category
        ]


        insights.append(
            f"You spend the most on {top_category} (₹{top_amount:,.0f})."
        )


        # WEEKDAY VS WEEKEND

        weekend_total = Decimal(0)
        weekday_total = Decimal(0)

        for expense in expenses:

            if expense.date.weekday() >= 5:

                weekend_total += expense.amount

            else:

                weekday_total += expense.amount


        if weekend_total > weekday_total:

            insights.append(
                "Your spending is higher during weekends."
            )

        else:

            insights.append(
                "Your spending is higher during weekdays."
            )


        # EVENING SPENDING

        evening_expenses = Decimal(0)

        for expense in expenses:

            if expense.date.hour >= 18:

                evening_expenses += expense.amount


        if evening_expenses > (total_spending * Decimal('0.4')):

            insights.append(
                "A significant portion of your expenses occur during evenings."
            )


        # HIGH VALUE EXPENSES

        average_expense = (
            total_spending / expenses.count()
        )


        large_expenses = [

            e for e in expenses

            if e.amount > average_expense * 3
        ]


        if large_expenses:

            insights.append(
                f"You made {len(large_expenses)} unusually high-value purchases."
            )


        return {

            "insights": insights,

            "total_spending": total_spending,

            "top_category": top_category,

            "top_category_amount": top_amount,
        }