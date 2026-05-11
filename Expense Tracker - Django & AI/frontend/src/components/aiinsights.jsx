import {
    Brain,
    Sparkles,
    TrendingUp,
    AlertTriangle,
    Wallet
} from "lucide-react";


function AIInsights({ insights }) {

    const getIcon = (text) => {

        const lower =
            text.toLowerCase();

        if (
            lower.includes("higher")
            || lower.includes("increase")
            || lower.includes("most")
        ) {

            return (
                <TrendingUp
                    className="text-emerald-400"
                    size={20}
                />
            );
        }


        if (
            lower.includes("unusual")
            || lower.includes("high-value")
            || lower.includes("risk")
        ) {

            return (
                <AlertTriangle
                    className="text-amber-400"
                    size={20}
                />
            );
        }


        return (
            <Wallet
                className="text-violet-400"
                size={20}
            />
        );
    };


    return (

        <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl">

            {/* HEADER */}

            <div className="flex items-center justify-between mb-10">

                <div className="flex items-center gap-5">

                    <div className="bg-violet-500/10 border border-violet-500/20 p-4 rounded-2xl">

                        <Brain
                            className="text-violet-400"
                            size={30}
                        />

                    </div>


                    <div>

                        <h2 className="text-3xl font-bold text-white">
                            AI Financial Insights
                        </h2>

                        <p className="text-slate-400 mt-2 text-lg">
                            Personalized intelligent spending analysis
                        </p>

                    </div>

                </div>


                <div className="hidden md:flex items-center gap-2 bg-violet-500/10 border border-violet-500/20 px-4 py-2 rounded-2xl">

                    <Sparkles
                        className="text-violet-400"
                        size={18}
                    />

                    <span className="text-violet-300 text-sm font-medium">
                        AI Powered
                    </span>

                </div>

            </div>


            {/* INSIGHTS */}

            {insights.length === 0 ? (

                <div className="bg-slate-800/50 border border-slate-700 rounded-3xl p-10 text-center">

                    <Brain
                        className="mx-auto text-slate-500 mb-4"
                        size={40}
                    />

                    <h3 className="text-xl text-white font-semibold">
                        No Insights Yet
                    </h3>

                    <p className="text-slate-400 mt-2">
                        Add more expenses to generate intelligent financial insights.
                    </p>

                </div>

            ) : (

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

    {insights.map(
        (insight, index) => (

        <div
            key={index}
            className="relative overflow-hidden bg-gradient-to-br from-slate-800 via-slate-850 to-slate-900 border border-slate-700/70 hover:border-violet-500/30 rounded-[28px] p-6 transition-all duration-300 hover:scale-[1.015] hover:shadow-2xl hover:shadow-violet-500/5"
        >

            {/* GLOW */}

            <div className="absolute top-0 right-0 h-32 w-32 bg-violet-500/5 blur-3xl rounded-full" />


            <div className="relative flex flex-col h-full">

                {/* TOP */}

                <div className="flex items-start gap-4">

                    {/* ICON */}

                    <div className="bg-slate-900 border border-slate-700 h-14 w-14 rounded-2xl flex items-center justify-center shrink-0 shadow-inner">

                        {getIcon(insight)}

                    </div>


                    {/* TEXT */}

                    <div className="flex-1">

                        <p className="text-slate-100 leading-relaxed text-[18px] font-medium">

                            {insight}

                        </p>

                    </div>

                </div>


                {/* FOOTER */}

                <div className="mt-6 pt-5 border-t border-slate-700/60 flex items-center gap-3">

                    <div className="h-2 w-2 rounded-full bg-violet-400 animate-pulse" />

                    <span className="text-slate-400 text-sm">

                        AI-generated insight

                    </span>

                </div>

            </div>

        </div>

    ))}

</div>

            )}

        </div>
    );
}

export default AIInsights;