import { useLocation, useNavigate } from "react-router-dom";
import Button from "../components/ui/Button";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import DownloadButton from "../components/ui/DownloadButton";
import { useMemo, useState } from "react";
import Papa from "papaparse";

export default function AnalysisResultPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const csvData = location.state?.csvData as string;

    const [currentPage, setCurrentPage] = useState(1);
    const [rowsPerPage] = useState(10);

    if (!csvData) {
        return (
            <div className="p-8 flex flex-col items-center">
                <h1 className="text-2xl font-bold text-red-500 mb-4">No Data Found</h1>
                <Button onClick={() => navigate("/csv-upload")}>Go Back</Button>
            </div>
        );
    }

    // Simple CSV Parser
    const result = Papa.parse(csvData, {
        header: true,
        skipEmptyLines: true,
    });

    const headers = (result.meta.fields || []) as string[];
    const data = result.data as Record<string, string>[];

    const totalPages = Math.ceil(data.length / rowsPerPage);

    const currentTableData = useMemo(() => {
        const firstPageIndex = (currentPage - 1) * rowsPerPage;
        const lastPageIndex = firstPageIndex + rowsPerPage;

        // Slice ข้อมูลเฉพาะส่วนที่ต้องแสดง
        return data.slice(firstPageIndex, lastPageIndex);
    }, [data, currentPage, rowsPerPage]);

    const goToNextPage = () => {
        if (currentPage < totalPages) {
            setCurrentPage((prev) => prev + 1);
        }
    };

    const goToPrevPage = () => {
        if (currentPage > 1) {
            setCurrentPage((prev) => prev - 1);
        }
    };

    const startIndex = (currentPage - 1) * rowsPerPage + 1;
    const endIndex = Math.min(currentPage * rowsPerPage, data.length);

    return (
        <div className="p-8 h-screen bg-gray-100 overflow-auto">
            <div className="max-w-7xl mx-auto bg-white shadow-lg rounded-xl p-8">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-4">
                        <button
                            onClick={() => navigate("/csv-upload")}
                            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                        >
                            <ArrowLeft className="h-6 w-6 text-gray-600" />
                        </button>
                        <h1 className="text-2xl font-bold text-gray-800">Analysis Results</h1>
                    </div>
                    <div className="w-1/2 flex justify-end space-x-4">
                        <Button className="p-4" onClick={() => navigate("/csv-upload")}>Upload New File</Button>
                        <DownloadButton csvData={csvData} />
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-100">
                            <tr>
                                {headers.map((header, index) => (
                                    <th
                                        key={index}
                                        scope="col"
                                        className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
                                    >
                                        {header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {currentTableData.map((row: any, rowIndex: number) => (
                                <tr key={rowIndex} className="hover:bg-gray-50">
                                    {headers.map((header, cellIndex) => (
                                        <td
                                            key={cellIndex}
                                            className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 "
                                        >
                                            {row[header]}
                                        </td>

                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                {data.length > rowsPerPage && (
                    <div className="flex items-center justify-between border-t border-gray-200 bg-white px-4 py-3 sm:px-6 mt-4">
                        <div className="flex flex-1 justify-between sm:hidden">
                            <button onClick={goToPrevPage} disabled={currentPage === 1} className="relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">Previous</button>
                            <button onClick={goToNextPage} disabled={currentPage === totalPages} className="relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">Next</button>
                        </div>
                        <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
                            <div>
                                <p className="text-sm text-gray-700">
                                    Showing <span className="font-medium">{startIndex}</span> to <span className="font-medium">{endIndex}</span> of{' '}
                                    <span className="font-medium">{data.length}</span> results
                                </p>
                            </div>
                            <div>
                                <nav className="isolate inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
                                    <button
                                        onClick={goToPrevPage}
                                        disabled={currentPage === 1}
                                        className="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <ChevronLeft className="h-5 w-5" aria-hidden="true" />
                                    </button>

                                    <span className="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 focus:z-20 focus:outline-offset-0">
                                        Page {currentPage} of {totalPages}
                                    </span>

                                    <button
                                        onClick={goToNextPage}
                                        disabled={currentPage === totalPages}
                                        className="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <ChevronRight className="h-5 w-5" aria-hidden="true" />
                                    </button>
                                </nav>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
