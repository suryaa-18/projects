import { Link } from "react-router-dom";

import {
    PiggyBank,
    ArrowRight
} from "lucide-react";


function PlanningCard() {

    return (

        <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl">

            <div className="flex items-center justify-between">

                <div>

                    <div className="bg-emerald-500/10 border border-emerald-500/20 h-16 w-16 rounded-2xl flex items-center justify-center mb-5">

                        <PiggyBank
                            className="text-emerald-400"
                            size={30}
                        />

                    </div>


                    <h2 className="text-3xl font-bold text-white">

                        Financial Planning

                    </h2>

                    <p className="text-slate-400 mt-3 max-w-md">

                        Manage budgets, savings goals,
                        and smart financial planning.

                    </p>

                </div>


                <Link
                    to="/planning"
                    className="bg-blue-600 hover:bg-blue-700 transition-all rounded-2xl px-6 py-4 text-white font-semibold flex items-center gap-3"
                >

                    Manage

                    <ArrowRight size={18} />

                </Link>

            </div>

        </div>
    );
}

export default PlanningCard;