#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <functional>

// Basic matrix class template
template <typename T>
class Matrix
{
private:
  std::vector<std::vector<T>> data;
  size_t rows, cols;

public:
  Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<T>(c, 0)) {}

  // Initialize with random values
  void randomInit(T min = 0, T max = 100)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    for (auto &row : data)
    {
      for (T &val : row)
      {
        val = dis(gen);
      }
    }
  }

  // Traditional serial matrix multiplication
  Matrix<T> multiply_serial(const Matrix<T> &other) const
  {
    if (cols != other.rows)
    {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix<T> result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < other.cols; ++j)
      {
        for (size_t k = 0; k < cols; ++k)
        {
          result.data[i][j] += data[i][k] * other.data[k][j];
        }
      }
    }
    return result;
  }

  // Async matrix multiplication using std::async
  Matrix<T> multiply_async(const Matrix<T> &other) const
  {
    if (cols != other.rows)
    {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix<T> result(rows, other.cols);
    std::vector<std::future<void>> futures;

    // Parallelize row calculations
    for (size_t i = 0; i < rows; ++i)
    {
      futures.push_back(std::async(std::launch::async, [&, i]()
                                   {
                for (size_t j = 0; j < other.cols; ++j) {
                    for (size_t k = 0; k < cols; ++k) {
                        result.data[i][j] += data[i][k] * other.data[k][j];
                    }
                } }));
    }

    // Wait for all async tasks to complete
    for (auto &fut : futures)
    {
      fut.wait();
    }

    return result;
  }

  // Thread-based matrix multiplication
  Matrix<T> multiply_thread_pool(const Matrix<T> &other) const
  {
    if (cols != other.rows)
    {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix<T> result(rows, other.cols);
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Divide work among threads
    size_t chunk_size = rows / num_threads;
    size_t remainder = rows % num_threads;

    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t thread_rows = chunk_size + (t < remainder ? 1 : 0);

      threads.emplace_back([&, start, thread_rows]()
                           {
                for (size_t i = start; i < start + thread_rows; ++i) {
                    for (size_t j = 0; j < other.cols; ++j) {
                        for (size_t k = 0; k < cols; ++k) {
                            result.data[i][j] += data[i][k] * other.data[k][j];
                        }
                    }
                } });

      start += thread_rows;
    }

    // Join all threads
    for (auto &thread : threads)
    {
      thread.join();
    }

    return result;
  }

  // Benchmark different multiplication methods
  void benchmark(const Matrix<T> &other)
  {
    auto start = std::chrono::high_resolution_clock::now();
    multiply_serial(other);
    auto end = std::chrono::high_resolution_clock::now();
    auto serial_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Serial Multiplication Time: " << serial_duration.count() << " µs\n";

    start = std::chrono::high_resolution_clock::now();
    multiply_async(other);
    end = std::chrono::high_resolution_clock::now();
    auto async_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Async Multiplication Time: " << async_duration.count() << " µs\n";

    start = std::chrono::high_resolution_clock::now();
    multiply_thread_pool(other);
    end = std::chrono::high_resolution_clock::now();
    auto thread_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Thread Pool Multiplication Time: " << thread_duration.count() << " µs\n";
  }

  // Print matrix (for debugging)
  void print() const
  {
    for (const auto &row : data)
    {
      for (T val : row)
      {
        std::cout << val << " ";
      }
      std::cout << "\n";
    }
  }
};

int main()
{
  const size_t SIZE = 500; // Adjust size for performance testing

  Matrix<double> A(SIZE, SIZE);
  Matrix<double> B(SIZE, SIZE);

  A.randomInit();
  B.randomInit();

  A.benchmark(B);

  return 0;
}