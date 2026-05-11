import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import api from "../services/api";

function Signup() {

    const navigate = useNavigate();

    const [username, setUsername] = useState("");

    const [password, setPassword] = useState("");

    const handleSignup = async () => {

        if (!username.trim() || !password) {
            alert("Please enter both username and password.");
            return;
        }

        try {

            await api.post(
                "/api/signup/",
                {
                    username,
                    password
                }
            );

            alert("Account created successfully");

            navigate("/");

        } catch (error) {

            console.error(error);

            const message =
                error.response?.data?.error ||
                "Signup failed. Please try again.";

            alert(message);
        }
    };

    return (

        <div className="min-h-screen bg-slate-950 flex items-center justify-center px-6">

            <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-10 shadow-2xl">

                <div className="mb-8 text-center">

                    <h1 className="text-4xl font-bold text-white">
                        Create Account
                    </h1>

                    <p className="text-slate-400 mt-3">
                        Start managing your expenses intelligently
                    </p>

                </div>

                <div className="space-y-5">

                    <div>

                        <label className="block text-slate-300 mb-2 text-sm">
                            Username
                        </label>

                        <input
                            type="text"
                            placeholder="Choose username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none focus:border-blue-500"
                        />

                    </div>

                    <div>

                        <label className="block text-slate-300 mb-2 text-sm">
                            Password
                        </label>

                        <input
                            type="password"
                            placeholder="Create password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none focus:border-blue-500"
                        />

                    </div>

                </div>

                <button
                    onClick={handleSignup}
                    className="w-full mt-8 bg-blue-600 hover:bg-blue-700 transition-all rounded-xl py-3 font-semibold text-white"
                >
                    Create Account
                </button>

                <p className="text-center text-slate-400 mt-6">

                    Already have an account?{" "}

                    <Link
                        to="/"
                        className="text-blue-400 hover:text-blue-300"
                    >
                        Login
                    </Link>

                </p>

            </div>

        </div>
    );
}

export default Signup;