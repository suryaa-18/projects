import { useEffect, useMemo, useState } from "react";

import AIInsights from "../components/AIInsights";
import ExpensePieChart from "../components/ExpensePieChart";
import ExpenseList from "../components/ExpenseList";
import MonthlyTrendChart from "../components/MonthlyTrendChart";
import FinancialHealthCard from "../components/FinancialHealthCard";
import BudgetSection from "../components/BudgetSection";
import PlanningCard from "../components/PlanningCard";

import {
    Wallet,
    TrendingUp,
    Tags,
    LogOut,
    Sparkles,
    CalendarDays,
    Clock3
} from "lucide-react";

import {
    useNavigate,
    Link
} from "react-router-dom";

import api from "../services/api";


function Dashboard() {

    const navigate = useNavigate();

    const [text, setText] =
        useState("");

    const [expenses, setExpenses] =
        useState([]);

    const [loading, setLoading] =
        useState(false);

    const [successMessage,
        setSuccessMessage] =
        useState("");

    const [currentTime,
        setCurrentTime] =
        useState(new Date());

    const [insights,
        setInsights] =
        useState([]);

    const [financialHealth,
        setFinancialHealth] =
        useState(null);


    useEffect(() => {

        fetchExpenses();

        fetchInsights();

        fetchFinancialHealth();

    }, []);


    // LIVE CLOCK

    useEffect(() => {

        const timer = setInterval(() => {

            setCurrentTime(
                new Date()
            );

        }, 1000);

        return () =>
            clearInterval(timer);

    }, []);


    // FETCH INSIGHTS

    const fetchInsights =
        async () => {

        try {

            const response =
                await api.get(
                    "/api/expenses/insights/"
                );

            setInsights(
                response.data.insights
            );

        } catch (error) {

            console.error(error);
        }
    };


    // FETCH FINANCIAL HEALTH

    const fetchFinancialHealth =
        async () => {

        try {

            const response =
                await api.get(
                    "/api/expenses/financial-health/"
                );

            setFinancialHealth(
                response.data
            );

        } catch (error) {

            console.error(error);
        }
    };


    // FETCH EXPENSES

    const fetchExpenses =
        async () => {

        try {

            const response =
                await api.get(
                    "/api/expenses/"
                );

            setExpenses(
                response.data
            );

        } catch (error) {

            console.error(error);
        }
    };


    // ADD EXPENSE

    const handleSubmit =
        async () => {

        if (!text.trim()) return;

        try {

            setLoading(true);

            const response =
                await api.post(
                    "/ai/parse-expense/",
                    {
                        text
                    }
                );

            setText("");

            await fetchExpenses();

            const count =
                response.data.count
                || 1;

            setSuccessMessage(
                `${count} expense${
                    count > 1
                        ? "s"
                        : ""
                } added successfully`
            );

            setTimeout(() => {

                setSuccessMessage("");

            }, 2500);

        } catch (error) {

            console.error(error);

            alert(
                "Failed to add expense"
            );

        } finally {

            setLoading(false);
        }
    };


    // LOGOUT

    const handleLogout = () => {

        localStorage.removeItem(
            "token"
        );

        navigate("/");
    };


    // TOTAL EXPENSES

    const totalExpenses =
        useMemo(() => {

        return expenses.reduce(
            (sum, expense) =>
                sum +
                Number(
                    expense.amount
                ),
            0
        );

    }, [expenses]);


    // MONTHLY EXPENSES

    const monthlyExpenses =
        useMemo(() => {

        const currentMonth =
            new Date().getMonth();

        return expenses
            .filter(
                (expense) => {

                const expenseDate =
                    new Date(
                        expense.date
                    );

                return (
                    expenseDate.getMonth()
                    === currentMonth
                );
            })
            .reduce(
                (
                    sum,
                    expense
                ) =>
                    sum +
                    Number(
                        expense.amount
                    ),
                0
            );

    }, [expenses]);


    // TOP CATEGORY

    const topCategory =
        useMemo(() => {

        const counts = {};

        expenses.forEach(
            (expense) => {

            const category =
                expense.category_name
                || "Miscellaneous";

            counts[category] =
                (
                    counts[
                        category
                    ] || 0
                ) + 1;
        });

        let top = "None";

        let max = 0;

        for (
            const category
            in counts
        ) {

            if (
                counts[
                    category
                ] > max
            ) {

                max =
                    counts[
                        category
                    ];

                top =
                    category;
            }
        }

        return top;

    }, [expenses]);


    // MOST SPENT CATEGORY

    const mostSpentCategory =
        useMemo(() => {

        const totals = {};

        expenses.forEach(
            (expense) => {

            const category =
                expense.category_name
                || "Miscellaneous";

            const amount =
                Number(
                    expense.amount
                );

            totals[category] =
                (
                    totals[
                        category
                    ] || 0
                ) + amount;
        });

        let top = "None";

        let max = 0;

        for (
            const category
            in totals
        ) {

            if (
                totals[
                    category
                ] > max
            ) {

                max =
                    totals[
                        category
                    ];

                top =
                    category;
            }
        }

        return {
            category: top,
            amount: max
        };

    }, [expenses]);


    return (

        <div className="min-h-screen bg-slate-950 text-white">

            <div className="max-w-7xl mx-auto px-6 py-10">


                {/* HEADER */}

                <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6 mb-12">

                    <div>

                        <h1 className="text-5xl font-bold tracking-tight">

                            AI Expense Tracker

                        </h1>

                        <p className="text-slate-400 mt-3 text-lg">

                            Smart financial tracking powered by AI

                        </p>


                        {/* DATE & TIME */}

                        <div className="mt-6 flex flex-col gap-3">

                            <div className="flex items-center gap-3 text-slate-300">

                                <CalendarDays
                                    size={18}
                                />

                                <span>

                                    {currentTime.toLocaleDateString(
                                        "en-IN",
                                        {
                                            weekday:
                                                "long",
                                            year:
                                                "numeric",
                                            month:
                                                "long",
                                            day:
                                                "numeric"
                                        }
                                    )}

                                </span>

                            </div>


                            <div className="flex items-center gap-3 text-white text-xl font-semibold">

                                <Clock3
                                    size={20}
                                />

                                <span>

                                    {currentTime.toLocaleTimeString(
                                        "en-IN"
                                    )}

                                </span>

                            </div>

                        </div>

                    </div>


                    <button
                        onClick={
                            handleLogout
                        }
                        className="flex items-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 px-5 py-3 rounded-2xl transition-all h-fit"
                    >

                        <LogOut
                            size={18}
                        />

                        Logout

                    </button>

                </div>


                {/* ANALYTICS */}

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8 mb-14">

                    {/* TOTAL */}

                    <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6">

                        <div className="flex items-center justify-between">

                            <div>

                                <p className="text-slate-400 text-sm">

                                    Total Expenses

                                </p>

                                <h2 className="text-4xl font-bold mt-3">

                                    ₹{
                                        totalExpenses.toLocaleString()
                                    }

                                </h2>

                            </div>

                            <div className="bg-blue-500/10 p-4 rounded-2xl">

                                <Wallet
                                    className="text-blue-400"
                                    size={32}
                                />

                            </div>

                        </div>

                    </div>


                    {/* MONTHLY */}

                    <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6">

                        <div className="flex items-center justify-between">

                            <div>

                                <p className="text-slate-400 text-sm">

                                    Monthly Spending

                                </p>

                                <h2 className="text-4xl font-bold mt-3">

                                    ₹{
                                        monthlyExpenses.toLocaleString()
                                    }

                                </h2>

                            </div>

                            <div className="bg-green-500/10 p-4 rounded-2xl">

                                <TrendingUp
                                    className="text-green-400"
                                    size={32}
                                />

                            </div>

                        </div>

                    </div>


                    {/* TOP CATEGORY */}

                    <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6">

                        <div className="flex items-center justify-between">

                            <div>

                                <p className="text-slate-400 text-sm">

                                    Top Category

                                </p>

                                <h2 className="text-4xl font-bold mt-3">

                                    {topCategory}

                                </h2>

                            </div>

                            <div className="bg-purple-500/10 p-4 rounded-2xl">

                                <Tags
                                    className="text-purple-400"
                                    size={32}
                                />

                            </div>

                        </div>

                    </div>


                    {/* MOST SPENT */}

                    <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6">

                        <div className="flex items-center justify-between">

                            <div>

                                <p className="text-slate-400 text-sm">

                                    Most Spent Category

                                </p>

                                <h2 className="text-3xl font-bold mt-3">

                                    {
                                        mostSpentCategory.category
                                    }

                                </h2>

                                <p className="text-slate-400 mt-2">

                                    ₹{
                                        mostSpentCategory.amount
                                        .toLocaleString()
                                    }

                                </p>

                            </div>

                            <div className="bg-orange-500/10 p-4 rounded-2xl">

                                <Wallet
                                    className="text-orange-400"
                                    size={32}
                                />

                            </div>

                        </div>

                    </div>

                </div>


                {/* ADD EXPENSE */}

                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 mb-10">

                    <div className="flex items-center gap-3 mb-5">

                        <Sparkles
                            className="text-blue-400"
                            size={24}
                        />

                        <h2 className="text-2xl font-semibold">

                            Add Expense

                        </h2>

                    </div>

                    <textarea
                        rows="4"
                        placeholder="Try: I spent 300 on dinner, 600 on drinks and 1000 on uber"
                        value={text}
                        onChange={(e) =>
                            setText(
                                e.target.value
                            )
                        }
                        className="w-full bg-slate-800 border border-slate-700 rounded-2xl p-5 text-white outline-none focus:border-blue-500 resize-none"
                    />

                    <div className="flex items-center justify-between mt-5">

                        <p className="text-slate-500 text-sm">

                            Press Ctrl + Enter

                        </p>

                        <button
                            onClick={
                                handleSubmit
                            }
                            disabled={
                                loading
                            }
                            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-all px-8 py-3 rounded-2xl font-semibold"
                        >

                            {
                                loading
                                    ? "Parsing..."
                                    : "Add Expense"
                            }

                        </button>

                    </div>

                    {successMessage && (

                        <div className="mt-5 bg-green-500/10 border border-green-500/20 text-green-400 rounded-2xl px-5 py-4">

                            {
                                successMessage
                            }

                        </div>
                    )}

                </div>


                {/* PIE CHART */}

                <div className="mb-10">

                    <ExpensePieChart
                        expenses={expenses}
                    />

                </div>


                {/* TREND + AI */}

                <div className="grid grid-cols-1 xl:grid-cols-2 gap-10 mb-10">

                    <MonthlyTrendChart
                        expenses={expenses}
                    />

                    <AIInsights
                        insights={insights}
                    />

                </div>


                {/* FINANCIAL HEALTH + PLANNING */}

                <div className="flex flex-col gap-10 mb-10">

                    <div className="w-full">
                        <FinancialHealthCard
                            data={financialHealth}
                        />
                    </div>

                    <div className="w-full">
                        <PlanningCard />
                    </div>

                </div>


                {/* EXPENSE LIST */}

                {/* RECENT EXPENSES */}

<div className="bg-slate-900 border border-slate-800 rounded-3xl p-8">

    <div className="flex items-center justify-between mb-8">

        <div>

            <h2 className="text-3xl font-bold text-white">

                Recent Expenses

            </h2>

            <p className="text-slate-400 mt-2">

                Showing latest 5 expenses

            </p>

        </div>


        <Link
            to="/expenses"
            className="bg-blue-600 hover:bg-blue-700 transition-all px-5 py-3 rounded-2xl text-white font-semibold"
        >

            Show All

        </Link>

    </div>


    <ExpenseList
        expenses={
            expenses.slice(0, 5)
        }
        fetchExpenses={
            fetchExpenses
        }
    />

</div>

            </div>

        </div>
    );
}

export default Dashboard;