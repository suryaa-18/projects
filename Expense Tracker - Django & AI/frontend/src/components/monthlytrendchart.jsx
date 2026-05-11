import {
    useMemo,
    useState,
    useEffect
} from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer
} from "recharts";

import {
    TrendingUp,
    CalendarDays
} from "lucide-react";


function MonthlyTrendChart({ expenses }) {

    // AVAILABLE MONTHS

    const availableMonths = useMemo(() => {

        const months = new Set();

        expenses.forEach((expense) => {

            const date =
                new Date(expense.date);

            const month =
                date.toLocaleString(
                    "default",
                    {
                        month: "long",
                        year: "numeric"
                    }
                );

            months.add(month);
        });

        return Array.from(months);

        }, [expenses]);
        useEffect(() => {

        if (
            availableMonths.length > 0
            && !selectedMonth
        ) {

            setSelectedMonth(
                availableMonths[
                    availableMonths.length - 1
                ]
            );
        }

    }, [availableMonths]);

    // DEFAULT MONTH

    const [selectedMonth,
    setSelectedMonth] =
    useState("");


    // FILTERED EXPENSES

    const filteredExpenses =
        expenses.filter((expense) => {

        const date =
            new Date(expense.date);

        const expenseMonth =
            date.toLocaleString(
                "default",
                {
                    month: "long",
                    year: "numeric"
                }
            );

        return (
            expenseMonth
            === selectedMonth
        );
    });


    // DAILY TOTALS

    const dailyTotals = {};

    filteredExpenses.forEach((expense) => {

        const date =
            new Date(expense.date);

        const formattedDate =
            date.toLocaleDateString(
                "en-IN",
                {
                    day: "numeric",
                    month: "short"
                }
            );

        dailyTotals[formattedDate] =
            (dailyTotals[
                formattedDate
            ] || 0)
            + Number(expense.amount);
    });


    // CHART DATA

    const chartData =
        Object.entries(dailyTotals)
        .map(([date, total]) => ({
            date,
            total
        }))
        .sort((a, b) => {

            const parseDate =
                (str) => {

                return new Date(
                    `${str} 2026`
                );
            };

            return (
                parseDate(a.date)
                - parseDate(b.date)
            );
        });


    return (

        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-xl">

            {/* HEADER */}

            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-5 mb-8">

                <div className="flex items-center gap-4">

                    <div className="bg-green-500/10 p-3 rounded-2xl">

                        <TrendingUp
                            className="text-green-400"
                            size={24}
                        />

                    </div>

                    <div>

                        <h2 className="text-2xl font-semibold text-white">
                            Daily Expense Trend
                        </h2>

                        <p className="text-slate-400 mt-1">
                            Daily spending pattern analysis
                        </p>

                    </div>

                </div>


                {/* MONTH SELECTOR */}

                <div className="flex items-center gap-3 bg-slate-800 border border-slate-700 rounded-2xl px-4 py-3">

                    <CalendarDays
                        className="text-slate-400"
                        size={18}
                    />

                    <select
                        value={selectedMonth}
                        onChange={(e) =>
                            setSelectedMonth(
                                e.target.value
                            )
                        }
                        className="bg-transparent text-white outline-none"
                    >

                        {availableMonths.map(
                            (month, index) => (

                            <option
                                key={index}
                                value={month}
                                className="bg-slate-900"
                            >

                                {month}

                            </option>

                        ))}

                    </select>

                </div>

            </div>


            {/* CHART */}

            <div className="h-[420px]">

                <ResponsiveContainer
                    width="100%"
                    height="100%"
                >

                    <LineChart
                        data={chartData}
                        margin={{
                            top: 10,
                            right: 20,
                            left: 10,
                            bottom: 10
                        }}
                    >

                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#334155"
                        />

                        <XAxis
                            dataKey="date"
                            stroke="#94A3B8"
                        />

                        <YAxis
                            stroke="#94A3B8"
                        />

                        <Tooltip
                            contentStyle={{
                                backgroundColor:
                                    "#0f172a",
                                border:
                                    "1px solid #334155",
                                borderRadius:
                                    "16px",
                                color:
                                    "white"
                            }}
                            formatter={(value) =>
                                `₹${Number(
                                    value
                                ).toLocaleString()}`
                            }
                        />

                        <Line
                            type="monotone"
                            dataKey="total"
                            stroke="#22C55E"
                            strokeWidth={4}
                            dot={{
                                r: 5
                            }}
                            activeDot={{
                                r: 8
                            }}
                        />

                    </LineChart>

                </ResponsiveContainer>

            </div>

        </div>
    );
}

export default MonthlyTrendChart;