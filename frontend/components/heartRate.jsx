// pages/index.jsx
import { useEffect, useRef, useState, useMemo } from "react";
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

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);

const socket = io("http://localhost:5000");

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [labels, setLabels] = useState([]);
  const [bpmData, setBpmData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    });

    // Send frame every 200ms
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

    // Receive BPM
    socket.on("bpm", (data) => {
      setLabels((prev) => [...prev.slice(-19), new Date().toLocaleTimeString()]);
      setBpmData((prev) => [...prev.slice(-19), data.bpm]);
      setIsLoading(false);
    });

    return () => {
      clearInterval(interval);
      socket.off("bpm");
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
      },
    ],
  };

  const stats = useMemo(() => {
    if (bpmData.length === 0) return null;
    const min = Math.min(...bpmData);
    const max = Math.max(...bpmData);
    const avg = bpmData.reduce((a, b) => a + b, 0) / bpmData.length;
    return { min, max, avg: avg.toFixed(1) };
  }, [bpmData]);

  const exportCSV = () => {
    const rows = ["Time,BPM", ...labels.map((label, i) => `${label},${bpmData[i]}`)];
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

  return (
    <main className="h-screen bg-gray-100 p-4">
      <div className="grid grid-cols-10 gap-4 h-full">
        {/* Webcam */}
        <div className="col-span-3 bg-white shadow-lg rounded-lg p-4 flex flex-col items-center justify-center">
          <video ref={videoRef} width="100%" className="rounded-lg" />
          <canvas ref={canvasRef} width={320} height={240} className="hidden" />
        </div>

        {/* Chart & Stats */}
        <div className="col-span-7 bg-white shadow-lg rounded-lg p-6 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Live Heart Rate Monitor</h2>
            <div className="space-x-2">
              <button
                onClick={exportCSV}
                className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded"
              >
                Export CSV
              </button>
              <button
                onClick={downloadChart}
                className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded"
              >
                Download Chart
              </button>
            </div>
          </div>

          {isLoading ? (
            <div className="flex-1 flex items-center justify-center text-gray-500">
              Waiting for BPM data...
            </div>
          ) : (
            <>
              <Line data={chartData} />
              {stats && (
                <div className="mt-4 flex justify-around text-sm text-gray-600">
                  <div>Min: {stats.min} BPM</div>
                  <div>Max: {stats.max} BPM</div>
                  <div>Avg: {stats.avg} BPM</div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </main>
  );
}

