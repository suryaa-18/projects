import {
    PieChart,
    Pie,
    Cell,
    Tooltip,
    ResponsiveContainer
} from "recharts";

import {
    PieChart as PieChartIcon
} from "lucide-react";


const COLORS = [
    "#3B82F6",
    "#10B981",
    "#8B5CF6",
    "#F59E0B",
    "#EF4444",
    "#14B8A6",
    "#EC4899",
    "#6366F1",
    "#84CC16",
    "#F97316",
];


function ExpensePieChart({ expenses }) {

    const categoryTotals = {};

    expenses.forEach((expense) => {

        const category =
            expense.category_name
            || "Miscellaneous";

        const amount =
            Number(expense.amount);

        categoryTotals[category] =
            (categoryTotals[category] || 0)
            + amount;
    });


    const chartData =
        Object.entries(categoryTotals)
        .map(([name, value]) => ({
            name,
            value
        }));


    const totalAmount =
        chartData.reduce(
            (sum, item) =>
                sum + item.value,
            0
        );


    return (

        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-xl">

            {/* HEADER */}

            <div className="flex items-center justify-between mb-10">

                <div className="flex items-center gap-4">

                    <div className="bg-blue-500/10 p-3 rounded-2xl">

                        <PieChartIcon
                            className="text-blue-400"
                            size={24}
                        />

                    </div>

                    <div>

                        <h2 className="text-2xl font-semibold text-white">
                            Expense Categories
                        </h2>

                        <p className="text-slate-400 mt-1">
                            Category-wise spending analysis
                        </p>

                    </div>

                </div>


                <div className="text-right">

                    <p className="text-slate-400 text-sm">
                        Total Expenses
                    </p>

                    <h2 className="text-3xl font-bold text-white mt-1">
                        ₹{totalAmount.toLocaleString()}
                    </h2>

                </div>

            </div>


            {/* CHART + LEGEND */}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">

                {/* PIE CHART */}

                <div className="h-[420px]">

                    <ResponsiveContainer
                        width="100%"
                        height="100%"
                    >

                        <PieChart>

                            <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={90}
                                outerRadius={170}
                                paddingAngle={3}
                                dataKey="value"
                                animationDuration={700}
                            >

                                {chartData.map(
                                    (entry, index) => (

                                    <Cell
                                        key={index}
                                        fill={
                                            COLORS[
                                                index
                                                % COLORS.length
                                            ]
                                        }
                                        stroke="#0f172a"
                                        strokeWidth={2}
                                    />

                                ))}

                            </Pie>


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

                        </PieChart>

                    </ResponsiveContainer>

                </div>


                {/* CUSTOM LEGEND */}

                <div className="space-y-5">

                    {chartData.map(
                        (item, index) => {

                        const percentage =
                            (
                                (item.value
                                / totalAmount)
                                * 100
                            ).toFixed(1);

                        return (

                            <div
                                key={index}
                                className="flex items-center justify-between bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4"
                            >

                                <div className="flex items-center gap-4">

                                    <div
                                        className="w-4 h-4 rounded-full"
                                        style={{
                                            backgroundColor:
                                                COLORS[
                                                    index
                                                    % COLORS.length
                                                ]
                                        }}
                                    />

                                    <span className="text-white text-lg">

                                        {item.name}

                                    </span>

                                </div>


                                <div className="text-right">

                                    <p className="text-white font-semibold">

                                        {percentage}%

                                    </p>

                                    <p className="text-slate-400 text-sm">

                                        ₹{
                                            item.value.toLocaleString()
                                        }

                                    </p>

                                </div>

                            </div>

                        );
                    })}

                </div>

            </div>

        </div>
    );
}

export default ExpensePieChart;