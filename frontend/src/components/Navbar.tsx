import { useNavigate } from "react-router-dom";
import logo from "../assets/logo.png";


export default function Navbar() {

  const navigate = useNavigate();
  return (
    <nav className="w-full shadow-sm ">
      <div className="grid grid-cols-12 gap-4 bg-white py-4">
        <div className="flex items-center col-start-3 col-span-2 space-x-4">
          <img src={logo} className="size-8"/>
          <p onClick={() => navigate("/")} className="text-xl text-gray-800 cursor-pointer ">Mobile Sentiment</p>
        </div>
        <div className="col-start-10  flex justify-center items-center space-x-4 text-gray-600 cursor-pointer " onClick={() => navigate("/dashboard")}>
          <p> Dashboard</p>
        </div>
        
      </div>
      
    </nav>
  );
}