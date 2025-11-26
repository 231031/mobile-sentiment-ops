
export default function Footer() {
    return (
        <footer className="fixed bottom-0 w-full bg-white">
            <div className=" mx-auto px-4 py-3">
                <div className="flex flex-col md:flex-row justify-center space-x-8 items-center text-xs ">
                    
                    {/* Left: Brand */}
                    <div className="mb-2 md:mb-0 ">
                        <span className="font-bold text-gray-800 text-sm">Mobile Sentiment</span>
                        <span className="text-orange-500 ">© 2025</span>
                    </div>

                    {/* Center: Team */}
                    <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-gray-600 ">
                        <div className="text-center group">
                            <p className="font-bold text-orange-500 mb-0.5  p-1">Data Scientist</p>
                            <p className="group-hover:text-gray-900 transition-colors p-1">6570501057 SIRIYAKORN KHIAOWIJIT</p>
                        </div>
                        <div className="text-center group">
                            <p className="font-bold text-orange-500 mb-0.5 p-1">ML Engineer</p>
                            <p className="group-hover:text-gray-900 transition-colors p-1">6570501037 PAWEEKORN SORATYATHORN</p>
                        </div>
                        <div className="text-center group">
                            <p className="font-bold text-orange-500 mb-0.5 p-1">Automation</p>
                            <p className="group-hover:text-gray-900 transition-colors p-1">6570501087 SUNEENAD SANGUANIN</p>
                        </div>
                        <div className="text-center group">
                            <p className="font-bold text-orange-500 mb-0.5 p-1">ML Infrastructure</p>
                            <p className="group-hover:text-gray-900 transition-colors p-1">6570501061 ARTHIT NOPJAROONSRI</p>
                            <p className="group-hover:text-gray-900 transition-colors p-1">6570501066 KANLAYAPHAT PRAKOBWAITAYAKIJ</p>
                        </div>
                    </div>

                    {/* Right: Policy */}
                    <div className="mt-2 md:mt-0">
                        <a href="#" className="text-gray-400 hover:text-orange-500 transition-colors font-medium">Privacy Policy</a>
                    </div>
                </div>
            </div>
        </footer>
    )
}