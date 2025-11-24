import React from "react"
import Navbar from "../components/Navbar"
import Footer from "../components/Footer"


export default function AppLayOut({children}: {children: React.ReactNode}) {
    return (
        <div className="app-container font-google-sans-flex">
            <Navbar/>
            <main className="app-content">
                {children}
            </main>
            <Footer/>
        </div>
    )
}