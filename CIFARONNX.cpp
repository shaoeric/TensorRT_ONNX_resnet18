// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"


#define HEIGHT 224
#define WIDTH 224
#define CHANNEL 3
#define BATCH_SIZE 1
#define NUM_CLASSES 1000
#define RUN_FP16 false
#define RUN_INT8 false

using namespace std;



class CIFARONNX
{
public:
	CIFARONNX(const string& onnx_file, const string& engine_file) : m_onnx_file(onnx_file), m_engine_file(engine_file) {};
	vector<float> prepareImage(const cv::Mat& img);
	bool onnxToTRTModel(nvinfer1::IHostMemory* trt_model_stream);
	bool loadEngineFromFile();
	void doInference(const cv::Mat& img);
private:
	const string m_onnx_file;
	const string m_engine_file;
	samplesCommon::Args gArgs;
	nvinfer1::ICudaEngine* m_engine;

	bool constructNetwork(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, nvinfer1::IBuilderConfig* config, nvonnxparser::IParser* parser);
	
	bool saveEngineFile(nvinfer1::IHostMemory* data);
	std::unique_ptr<char[]> readEngineFile(int& length);
	
	int64_t volume(const nvinfer1::Dims &d);
	unsigned int getElementSize(nvinfer1::DataType t);
};


vector<float> CIFARONNX::prepareImage(const cv::Mat& img) {
	int c = CHANNEL;
	int h = HEIGHT;
	int w = WIDTH;

	auto scaleSize = cv::Size(w, h);

	cv::Mat rgb;
	cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
	cv::Mat resized;
	cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);

	cv::Mat img_float;
	resized.convertTo(img_float, CV_32FC3, 1.f / 255.0);

	//HWC TO CHW
	vector<cv::Mat> input_channels(c);
	cv::split(img_float, input_channels);

	vector<float> result(h * w * c);
	auto data = result.data();
	int channel_length = h * w;
	for (int i = 0; i < c; ++i) {
		memcpy(data, input_channels[i].data, channel_length * sizeof(float));
		data += channel_length;  // 指针后移channel_length个单位
	}
	return result;
}


bool CIFARONNX::constructNetwork(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, nvinfer1::IBuilderConfig * config, nvonnxparser::IParser * parser)
{
	// 解析onnx文件
	if (!parser->parseFromFile(this->m_onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError << "Fail to parse ONNX file" << std::endl;
		return false;
	}

	// build the Engine
	builder->setMaxBatchSize(BATCH_SIZE);
	config->setMaxWorkspaceSize(1 << 30);
	if (RUN_FP16)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (RUN_INT8)
	{
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder, config, gArgs.useDLACore);
	return true;
}

// 保存plan文件数据
bool CIFARONNX::saveEngineFile(nvinfer1::IHostMemory * data)
{
	std::ofstream file;
	file.open(m_engine_file, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());
	cout << "save engine file done" << endl;
	file.close();
	return true;
}

// 从plan文件读取数据
std::unique_ptr<char[]> CIFARONNX::readEngineFile(int &length)
{
	ifstream file;
	file.open(m_engine_file, std::ios::in | std::ios::binary);
	// 获得文件流的长度
	file.seekg(0, std::ios::end);  // 把指针移到末尾
	length = file.tellg();  // 返回当前指针位置
	// 指针移到开始
	file.seekg(0, std::ios::beg);
	// 定义缓存
	std::unique_ptr<char[]> data(new char[length]);
	// 读取文件到缓存区
	file.read(data.get(), length);
	file.close();
	return data;
}

// 累积乘法 对binding的维度累乘 (3,224,224) => 3*224*224
inline int64_t CIFARONNX::volume(const nvinfer1::Dims & d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int CIFARONNX::getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
		case nvinfer1::DataType::kINT32: return 4;
		case nvinfer1::DataType::kFLOAT: return 4;
		case nvinfer1::DataType::kHALF: return 2;
		case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}


// 读取plan文件数据，构建engine
bool CIFARONNX::loadEngineFromFile()
{
	int length = 0; // 记录data的长度
	std::unique_ptr<char[]> data = readEngineFile(length);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
	nvonnxparser::IPluginFactory* onnx_plugin = nvonnxparser::createPluginFactory(gLogger.getTRTLogger());
	m_engine = runtime->deserializeCudaEngine(data.get(), length, onnx_plugin);
	if (!m_engine)
	{
		std::cout << "Failed to create engine" << std::endl;
		return false;
	}
	return true;
}

void CIFARONNX::doInference(const cv::Mat &img)
{
	nvinfer1::IExecutionContext* context = m_engine->createExecutionContext();
	assert(context != nullptr);
	int nbBindings = m_engine->getNbBindings();
	assert(nbBindings == 2);  // 输入和输出，一共是2个

	// 为输入和输出创建空间
	void* buffers[2];  // 待创建的空间  为指针数组
	std::vector<int64_t> buffer_size;  // 要创建的空间大小
	buffer_size.resize(nbBindings);
	for (int i = 0; i < nbBindings; i++)
	{
		nvinfer1::Dims dims = m_engine->getBindingDimensions(i); // (3, 224, 224)  (1000)
		nvinfer1::DataType dtype = m_engine->getBindingDataType(i); // 0, 0 也就是两个都是kFloat类型
		//std::cout << static_cast<int>(dtype) << endl;
		int64_t total_size = volume(dims) * 1 * getElementSize(dtype);
		buffer_size[i] = total_size;
		CHECK(cudaMalloc(&buffers[i], total_size));
	}
	
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream)); // 创建异步cuda流

	float * out = new float[NUM_CLASSES]; 

	// 开始推理
	auto t_start = std::chrono::high_resolution_clock::now();
	vector<float> cur_input = prepareImage(img);
	auto t_end = std::chrono::high_resolution_clock::now();
	float duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "loading image takes " << duration << "ms" << std::endl;
	if (!cur_input.data())
	{
		std::cout << "failed to prepare image" << std::endl;
	}

	// 将输入传递到GPU
	CHECK(cudaMemcpyAsync(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice, stream));

	// 异步执行
	t_start = std::chrono::high_resolution_clock::now();
	context->execute(BATCH_SIZE, buffers);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "inference takes: " << duration << "ms" << std::endl;

	// 输出传回给CPU
	CHECK(cudaMemcpyAsync(out, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	//printPredictProb(out);
	int predict_label = max_element(out, out + NUM_CLASSES) - out;
	std::cout << predict_label << std::endl;

}


/*
 * 在没有trt engine plan文件的情况下，从onnx文件构建engine，然后序列化成engine plan文件
 */
bool CIFARONNX::onnxToTRTModel(nvinfer1::IHostMemory* trt_model_stream)
{
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);

	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

	// 构建网络
	if (!constructNetwork(builder, network, config, parser))
	{
		return false;
	}
	// 构建引擎
	m_engine = builder->buildEngineWithConfig(*network, *config);
	assert(m_engine != nullptr);
	// 验证网络构建正确
	assert(network->getNbInputs() == 1);
	assert(network->getInput(0)->getDimensions().nbDims == 4);
	assert(network->getNbOutputs() == 1);
	assert(network->getOutput(0)->getDimensions().nbDims == 1);

	// 序列化
	trt_model_stream = m_engine->serialize();
	nvinfer1::IHostMemory* data = m_engine->serialize();
	saveEngineFile(data);

	parser->destroy();
	network->destroy();
	builder->destroy();
	//m_engine->destroy();
}


int main()
{
	cv::Mat src = cv::imread("D:\\000148.jpg");
	string onnx_file = "./resnet18.onnx";
	string engine_file = "./resnet18_imagenet.trt";
	IHostMemory* trt_model_stream{ nullptr };

	CIFARONNX cifar_onnx(onnx_file, engine_file);
	vector<float> result = cifar_onnx.prepareImage(src);
	
	// 打开文件，是否存在engine plan
	fstream engine_reader;
	engine_reader.open(engine_file, std::ios::in);
	if (engine_reader)
	{
		std::cout << "found engine plan" << endl;
		cifar_onnx.loadEngineFromFile();
	}
	else
	{
		cifar_onnx.onnxToTRTModel(trt_model_stream);
	}
	cifar_onnx.doInference(src);
}
