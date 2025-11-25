import MetricCards from "../components/MetricCard";
import ReportCarousel from "../components/ReportCarousel";

export default function DashBoard() {
   const reportImages = [
      "eda_sentiment_brand_negative_20251125_151046.png",
      "eda_sentiment_brand_neutral_20251125_151047.png",
      "eda_sentiment_brand_positive_20251125_151046.png",
      "eda_sentiment_20251125_151042.png"
   ].map(img => `${import.meta.env.VITE_API_URL}/reports/${img}`);

   return (
    <div className="h-screen bg-gray-100 grid grid-cols-12 grid-rows-12 gap-4 py-10 ">
      <MetricCards />
      
      <div className="col-span-8 col-start-3 row-span-8 bg-white rounded-xl shadow-lg p-4 flex justify-center items-center">
         <ReportCarousel images={reportImages} />
      </div>
    </div>
  );
}