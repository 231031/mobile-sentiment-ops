import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Button from "./ui/Button";

interface ReportCarouselProps {
    images: string[];
}

export default function ReportCarousel({ images }: ReportCarouselProps) {
    const [currentIndex, setCurrentIndex] = useState(0);

    const prevImage = () => {
        setCurrentIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1));
    };

    const nextImage = () => {
        setCurrentIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1));
    };

    if (images.length === 0) {
        return <div className="text-gray-500">No images available</div>;
    }

    return (
        <div className="relative w-full h-full flex flex-col items-center justify-center group p-6">
            <div className="relative w-full h-full flex items-center justify-center overflow-hidden rounded-lg">
                <img
                    src={images[currentIndex]}
                    alt={`Report ${currentIndex + 1}`}
                    className="max-h-full max-w-full object-contain transition-opacity duration-300"
                />
            </div>

            {/* Navigation Buttons */}
            <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex justify-between px-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                <Button
                    onClick={prevImage}
                    className="pointer-events-auto bg-black/50 hover:bg-black/70 text-white rounded-full p-2"
                >
                    <ChevronLeft className="w-6 h-6" />
                </Button>
                <Button
                    onClick={nextImage}
                    className="pointer-events-auto bg-black/50 hover:bg-black/70 text-white rounded-full p-2"
                >
                    <ChevronRight className="w-6 h-6" />
                </Button>
            </div>

            {/* Indicators */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex space-x-2 bg-black/30 px-3 py-1 rounded-full">
                {images.map((_, index) => (
                    <div
                        key={index}
                        className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                            index === currentIndex ? "bg-white" : "bg-white/50"
                        }`}
                    />
                ))}
            </div>
            
            <div className="absolute top-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-xs">
                {currentIndex + 1} / {images.length}
            </div>
        </div>
    );
}
