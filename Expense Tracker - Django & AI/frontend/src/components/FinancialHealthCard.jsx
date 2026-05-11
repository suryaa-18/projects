import {
    ShieldCheck,
    Wallet,
    PiggyBank,
    TrendingUp
} from "lucide-react";


function FinancialHealthCard({

    data

}) {

    if (!data) return null;


    return (

        <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 rounded-[36px] p-8 shadow-2xl">

            {/* BACKGROUND GLOW */}

            <div className="absolute top-0 right-0 w-72 h-72 bg-emerald-500/5 blur-3xl rounded-full" />


            {/* HEADER */}

            <div className="relative flex flex-col xl:flex-row xl:items-center xl:justify-between gap-8 mb-10">

                <div>

                    <div className="flex items-center gap-5 mb-5">

                        <div className="bg-emerald-500/10 border border-emerald-500/20 h-20 w-20 rounded-3xl flex items-center justify-center shadow-lg">

                            <ShieldCheck
                                className="text-emerald-400"
                                size={38}
                            />

                        </div>


                        <div>

                            <h2 className="text-4xl font-bold text-white tracking-tight">

                                Financial Health

                            </h2>

                            <p className="text-slate-400 mt-2 text-lg">

                                AI-powered financial wellness overview

                            </p>

                        </div>

                    </div>

                </div>


                {/* HEALTH STATUS BADGE */}

                <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-3xl px-6 py-5 flex items-center gap-4 w-fit">

                    <TrendingUp
                        className="text-emerald-400"
                        size={26}
                    />

                    <div>

                        <p className="text-slate-400 text-sm">

                            Status

                        </p>

                        <h3 className="text-emerald-400 font-bold text-xl">

                            {data.health_status}

                        </h3>

                    </div>

                </div>

            </div>


            {/* STATS */}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-7">

                {/* HEALTH SCORE */}

                <div className="group relative bg-slate-800/80 backdrop-blur-xl border border-slate-700 hover:border-emerald-500/30 transition-all rounded-[32px] p-8 overflow-hidden">

                    <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 blur-2xl rounded-full" />

                    <p className="text-slate-400 text-base">

                        Health Score

                    </p>

                    <h2 className="text-7xl font-black text-white mt-6 leading-none">

                        {data.health_score}

                    </h2>

                    <div className="mt-6 flex items-center gap-3">

                        <div className="w-3 h-3 rounded-full bg-emerald-400" />

                        <span className="text-emerald-400 text-xl font-semibold">

                            {data.health_status}

                        </span>

                    </div>

                </div>


                {/* TOTAL BUDGET */}

                <div className="group relative bg-slate-800/80 backdrop-blur-xl border border-slate-700 hover:border-blue-500/30 transition-all rounded-[32px] p-8 overflow-hidden">

                    <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 blur-2xl rounded-full" />

                    <div className="flex items-center gap-4 mb-6">

                        <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-2xl">

                            <Wallet
                                className="text-blue-400"
                                size={28}
                            />

                        </div>

                        <p className="text-slate-300 text-xl font-medium">

                            Total Budget

                        </p>

                    </div>

                    <h2 className="text-6xl font-black text-white leading-none">

                        ₹{
                            Number(
                                data.total_budget || 0
                            ).toLocaleString()
                        }

                    </h2>

                    <p className="text-slate-500 mt-5 text-lg">

                        Monthly allocated spending

                    </p>

                </div>


                {/* SAVINGS */}

                <div className="group relative bg-slate-800/80 backdrop-blur-xl border border-slate-700 hover:border-pink-500/30 transition-all rounded-[32px] p-8 overflow-hidden">

                    <div className="absolute top-0 right-0 w-32 h-32 bg-pink-500/5 blur-2xl rounded-full" />

                    <div className="flex items-center gap-4 mb-6">

                        <div className="bg-pink-500/10 border border-pink-500/20 p-4 rounded-2xl">

                            <PiggyBank
                                className="text-pink-400"
                                size={28}
                            />

                        </div>

                        <p className="text-slate-300 text-xl font-medium">

                            Savings

                        </p>

                    </div>

                    <h2 className="text-6xl font-black text-white leading-none">

                        ₹{
                            Number(
                                data.savings || 0
                            ).toLocaleString()
                        }

                    </h2>

                    <p className="text-slate-500 mt-5 text-lg">

                        Current accumulated savings

                    </p>

                </div>

            </div>


            {/* NOTIFICATIONS */}

            {data.notifications?.length > 0 && (

                <div className="mt-10">

                    <h3 className="text-2xl font-bold text-white mb-5">

                        Smart Notifications

                    </h3>

                    <div className="space-y-4">

                        {data.notifications.map(
                            (
                                notification,
                                index
                            ) => (

                                <div
                                    key={index}
                                    className="bg-amber-500/10 border border-amber-500/20 rounded-2xl px-6 py-5 text-amber-300 text-lg"
                                >

                                    {notification}

                                </div>
                            )
                        )}

                    </div>

                </div>
            )}

        </div>
    );
}

export default FinancialHealthCard;