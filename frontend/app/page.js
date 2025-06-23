"use client";
import { useEffect, useRef, useState, useMemo } from "react";
import { AiOutlineWarning, AiOutlineLoading3Quarters } from "react-icons/ai";
import { FaHeartbeat, FaUserSlash, FaHistory } from "react-icons/fa";
import io from "socket.io-client";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
);

const socket = io("http://localhost:5000");

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [labels, setLabels] = useState([]);
  const [bpmData, setBpmData] = useState([]);
  const [status, setStatus] = useState("Waiting for face...");
  const [isLoading, setIsLoading] = useState(true);
  const [pastAverages, setPastAverages] = useState([]);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    });

    const interval = setInterval(() => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (canvas && video) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const imageData = canvas.toDataURL("image/jpeg");
          socket.emit("frame", { image: imageData });
        }
      }
    }, 200);

    socket.on("bpm", (data) => {
      if (data.bpm === 0) {
        setBpmData([]);
        setLabels([]);
      } else {
        setLabels((prev) => [
          ...prev.slice(-19),
          new Date().toLocaleTimeString(),
        ]);
        setBpmData((prev) => [...prev.slice(-19), data.bpm]);
        setIsLoading(false);
      }
    });

    socket.on("status", (data) => {
      if (data.state === "noface") {
        setStatus("🚫 No face detected");
        setBpmData([]);
        setLabels([]);
      } else if (data.state === "calibrating") {
        setStatus("🛠️ Calibrating... Please stay still");
      } else if (data.state === "live") {
        setStatus(`❤️ Live BPM: ${data.bpm}`);
      }
    });

    socket.on("session_end", (data) => {
      const avg = parseFloat(data.avg_bpm.toFixed(1));
      setPastAverages((prev) => [...prev.slice(-2), avg]);
    });

    return () => {
      clearInterval(interval);
      socket.off("bpm");
      socket.off("status");
      socket.off("session_end");
      if (videoRef.current) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      setStatus("Waiting for face...");
      setBpmData([]);
      setLabels([]);
    };
  }, []);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Heart Rate (BPM)",
        data: bpmData,
        borderColor: "rgb(239 68 68)",
        backgroundColor: "rgba(239, 68, 68, 0.5)",
        fill: false,
        tension: 0.4,
        pointBackgroundColor: "rgb(239 68 68)",
        pointBorderColor: "#fff",
        pointHoverRadius: 6,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 13,
        },
        padding: 10,
        displayColors: false,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          color: "#6b7280",
          maxRotation: 0,
          maxTicksLimit: 8,
        },
      },
      y: {
        min: Math.max(0, Math.min(...(bpmData.length ? bpmData : [60])) - 10),
        max: Math.max(...(bpmData.length ? bpmData : [100])) + 10,
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          color: "#6b7280",
          stepSize: 10,
        },
      },
    },
  };

  const stats = useMemo(() => {
    if (bpmData.length === 0) return null;
    const min = Math.min(...bpmData);
    const max = Math.max(...bpmData);
    const avg = bpmData.reduce((a, b) => a + b, 0) / bpmData.length;
    return { min, max, avg: avg.toFixed(1) };
  }, [bpmData]);

  const exportCSV = () => {
    const rows = [
      "Time,BPM",
      ...labels.map((label, i) => `${label},${bpmData[i]}`),
    ];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bpm_data.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadChart = () => {
    const chartCanvas = document.querySelector("canvas.chartjs-render-monitor");
    if (chartCanvas) {
      const link = document.createElement("a");
      link.href = chartCanvas.toDataURL("image/png");
      link.download = "bpm_chart.png";
      link.click();
    }
  };

  const renderStatus = () => {
    if (status.includes("No face")) {
      return (
        <div className="w-full p-3 flex items-center justify-center gap-2 bg-red-100 border border-red-300 rounded-xl text-red-700 font-semibold shadow-inner">
          <FaUserSlash className="text-lg" />
          <span className="text-sm">{status}</span>
        </div>
      );
    } else if (status.includes("Calibrating")) {
      return (
        <div className="w-full p-3 flex items-center justify-center gap-2 bg-yellow-100 border border-yellow-300 rounded-xl text-yellow-800 font-semibold shadow-inner animate-pulse">
          <AiOutlineLoading3Quarters className="animate-spin text-lg" />
          <span className="text-sm">{status}</span>
        </div>
      );
    } else if (status.includes("Live BPM")) {
      return (
        <div className="w-full p-3 flex items-center justify-center gap-2 bg-green-100 border border-green-300 rounded-xl text-green-800 font-semibold shadow-inner">
          <FaHeartbeat className="text-lg text-red-500" />
          <span className="text-sm">{status}</span>
        </div>
      );
    } else {
      return (
        <div className="w-full p-3 flex items-center justify-center gap-2 bg-gray-100 border border-gray-300 rounded-xl text-gray-700 font-medium shadow-inner">
          <AiOutlineWarning className="text-lg" />
          <span className="text-sm">{status}</span>
        </div>
      );
    }
  };

  return (
    <main className="h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 overflow-hidden">
      <div className="grid grid-cols-10 gap-4 h-full">
        {/* Webcam Section */}
        <div className="col-span-3 bg-white shadow-xl rounded-xl p-4 flex flex-col">
          <div className="relative w-full mb-4">
            <video 
              ref={videoRef} 
              width="100%" 
              className="rounded-lg shadow-md border border-gray-200"
            />
            <div className="absolute inset-0 border-4 border-transparent rounded-lg animate-pulse pointer-events-none"></div>
          </div>
          <canvas ref={canvasRef} width={320} height={240} className="hidden" />
          
          {renderStatus()}
          
          {/* Past Averages Section */}
          {pastAverages.length > 0 && (
            <div className="w-full mt-9 bg-gradient-to-br from-indigo-50 to-purple-50 p-4 rounded-xl border border-indigo-100 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <FaHistory className="text-indigo-600" />
                <h3 className="text-sm font-semibold text-indigo-800">
                  Previous Sessions Avg BPM
                </h3>
              </div>
              
              <div className="grid grid-cols-3 gap-3">
                {pastAverages
                  .slice()
                  .reverse()
                  .map((bpm, idx) => (
                    <div 
                      key={idx} 
                      className="bg-white rounded-lg p-3 shadow-sm border border-gray-100 flex flex-col items-center justify-center transition-all hover:shadow-md hover:border-indigo-200"
                    >
                      <div className="text-xs text-gray-500 mb-1">
                        Session {pastAverages.length - idx}
                      </div>
                      <div className="flex items-baseline gap-1">
                        <span className="text-xl font-bold text-indigo-700">
                          {bpm}
                        </span>
                        <span className="text-xs text-gray-500">BPM</span>
                      </div>
                      <div className="mt-1 w-10 h-1 bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full"></div>
                    </div>
                  ))}
              </div>
              
              <div className="mt-3 text-center">
                <span className="text-xs text-gray-500 inline-block px-2 py-1 bg-white rounded-full">
                  Latest sessions shown
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Chart Section */}
        <div className="col-span-7 bg-white shadow-xl rounded-xl p-6 flex flex-col h-full">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <FaHeartbeat className="text-red-500" />
              Live Heart Rate Monitor
            </h2>
            <div className="space-x-2 flex">
              <button
                onClick={exportCSV}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg shadow-sm transition duration-200 flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                Export CSV
              </button>
              <button
                onClick={downloadChart}
                className="bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg shadow-sm transition duration-200 flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                </svg>
                Save Chart
              </button>
            </div>
          </div>

          <div className="flex-1 min-h-0 flex flex-col">
            {isLoading ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
                <AiOutlineLoading3Quarters className="animate-spin text-3xl mb-3 text-indigo-500" />
                <p className="italic">Waiting for heart rate data...</p>
                <p className="text-xs mt-2 text-gray-400">Please position your face in the camera view</p>
              </div>
            ) : (
              <>
                <div className="h-[calc(100%-60px)]"> {/* Fixed height for chart */}
                  <Line data={chartData} options={chartOptions} />
                </div>
                {stats && (
                  <div className="mt-4 flex justify-around text-sm text-gray-700 font-semibold">
                    <div className="bg-red-50 px-4 py-2 rounded-lg flex items-center gap-2">
                      <span className="text-red-700">📉 Min:</span> 
                      <span className="font-mono">{stats.min} BPM</span>
                    </div>
                    <div className="bg-green-50 px-4 py-2 rounded-lg flex items-center gap-2">
                      <span className="text-green-700">📈 Max:</span> 
                      <span className="font-mono">{stats.max} BPM</span>
                    </div>
                    <div className="bg-indigo-50 px-4 py-2 rounded-lg flex items-center gap-2">
                      <span className="text-indigo-700">📊 Avg:</span> 
                      <span className="font-mono">{stats.avg} BPM</span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}