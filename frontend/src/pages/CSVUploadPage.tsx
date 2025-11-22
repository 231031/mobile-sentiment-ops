import icon from "../assets/react.svg";
import CsvUploadCard from "../components/CsvUploadCard";


export default function TextAnalysisPage() {
  return (
    <div className="h-screen bg-gray-100 grid grid-cols-12 gap-4 py-10 grid-rows-12">

      <CsvUploadCard />
      
    </div>
  );
}
