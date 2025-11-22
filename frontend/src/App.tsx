import { Routes, Route } from "react-router-dom";

import AppLayOut from "./layout/AppLayOut"

import HomePage from "./pages/HomePage";
import TextAnalysisPage from "./pages/TextAnalysisPage";
import CSVUploadPage from "./pages/CSVUploadPage";
import Dashboard from "./pages/Dashboard";



export default function App() {

  return (
    <AppLayOut>
      <Routes>
        <Route path="/" element={<HomePage/>}/>
        <Route path="/text-analysis" element={<TextAnalysisPage/>}/>
        <Route path="/csv-upload" element={<CSVUploadPage/>}/>
        <Route path="/dashboard" element={<Dashboard/>}/>
      </Routes>
    </AppLayOut>
  )
}
