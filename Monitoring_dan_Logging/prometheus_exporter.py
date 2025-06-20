from prometheus_client import start_http_server, Gauge, Counter
import time
import random

# Define Gauges (nilai bisa naik turun)
latency = Gauge('inference_latency_seconds', 'Latency of model inference')
error_rate = Gauge('inference_error_rate', 'Error rate of inference')
cpu_usage = Gauge('inference_cpu_usage_percent', 'CPU usage percentage during inference')
memory_usage = Gauge('inference_memory_usage_mb', 'Memory usage during inference in MB')
throughput = Gauge('inference_throughput_rps', 'Inference throughput in requests per second')
success_rate = Gauge('inference_success_rate', 'Success rate of inference')
queue_size = Gauge('inference_queue_size', 'Number of inference requests in queue')
model_version = Gauge('inference_model_version', 'Version number of deployed model (for example 1.1, 2.0, etc.)')
gpu_usage = Gauge('inference_gpu_usage_percent', 'GPU usage percentage during inference')  # Metrik baru

# Define Counters (nilai hanya naik)
request_count = Counter('inference_request_count', 'Total number of inference requests')


def simulate_metrics():
    while True:
        latency.set(random.uniform(0.1, 0.5))
        request_count.inc()
        is_error = random.choice([False, False, False, True])

        if is_error:
            error_rate.set(0.05)
            success_rate.set(0.95)
        else:
            error_rate.set(0.0)
            success_rate.set(1.0)

        cpu_usage.set(random.uniform(10, 50))
        memory_usage.set(random.uniform(200, 500))
        throughput.set(random.uniform(5, 20))
        queue_size.set(random.randint(0, 10))
        model_version.set(1.1)
        gpu_usage.set(random.uniform(0, 100))  # Simulasi GPU usage

        time.sleep(5)


if __name__ == "__main__":
    start_http_server(8000)
    simulate_metrics()
