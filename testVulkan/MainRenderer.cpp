#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <chrono>

#include <iostream> // for reporting and propagate errors
#include <fstream>
#include <stdexcept> // for reporting and propagate errors
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib> // provides the EXIT_SUCCESS and EXIT_FAILURE macros.
#include <cstdint>
#include <optional>
#include <set>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>

#include "imgui_impl_vulkan.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;	// Number of simultaneus frames we allow

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;	// if in release mode, don't check for validation layers
#else
const bool enableValidationLayers = true;	// if in debug mode, check for validation layers
#endif

// function to create the VkDebugUtilsMessengerEXT object 
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");	// look for the address and returns a nullptr if the function cannot be loaded
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);	// proxy function that handles the address issue
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;	// return and string to notify of an error with the extension
	}
}

// function to destroy the VkDebugUtilsMessengerEXT object 
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"); // look for the address and returns a nullptr if the function cannot be loaded
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

// It holds an index number for every queue in the system or nothing if it was not set up yet.
struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;	// queue to support graphics commands
	std::optional<uint32_t> presentFamily;

	bool isComplete() {	// ask whether every queue has a index or not.
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;	// number of images in swap change, width/height of images...
	std::vector<VkSurfaceFormatKHR> formats;	// pixel format, color space
	std::vector<VkPresentModeKHR> presentModes;	// the available presentation modes
};

// struct for all the information that every vertex will have
struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
	glm::vec3 normalCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, normalCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {	// overloaded operator of comparison in order to be able to selct which vertices are repeated
		return pos == other.pos && color == other.color && texCoord == other.texCoord && normalCoord == other.normalCoord;
	}
};

namespace std {
	template<> struct hash<Vertex> {	// hash function in order to be able to selct which vertices are repeated
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1)) >> 1) ^ (hash<glm::vec3>()(vertex.normalCoord) << 1);
		}
	};
}

struct UniformBufferObject {	// Union buffer with the information of the object
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct UniformBufferLight {	// Union buffer with the information of the light
	alignas(4) float specularHighlight;
	alignas(16) glm::vec3 ambientColor;
	alignas(16) glm::vec3 diffuseColor;
	alignas(16) glm::vec3 specularColor;
	alignas(16) glm::vec3 emissiveColor;
	alignas(16) glm::vec4 lightPosition;
};

struct UniformBufferShadow {	// Union buffer with the information of the object
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

// It will store the Vulkan objects that will be initialize, used and deallocated
class HelloTriangleApplication {
public:
	void run() {	// Abstaction of all the things that need to be done before doing anything in just one call to this method.
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;	// holds a pointer to the window

	VkInstance instance;	// holds the instance
	VkDebugUtilsMessengerEXT debugMessenger;	// handle debug messages
	VkSurfaceKHR surface;	// represents a surface to render images on it

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;	// Holds the graphics card selected and it will be destroyed when the VkInstance is destroyed.
	VkDevice device;	// holds the logical device handler

	VkQueue graphicsQueue;	// handler graphics queue
	VkQueue presentQueue;	// handler present queue

	VkSwapchainKHR swapChain;	// handler of the swap chain
	std::vector<VkImage> swapChainImages;	// Array containing all the images of the swap chain
	VkFormat swapChainImageFormat;	// Handler of the format of the swap chain
	VkExtent2D swapChainExtent;	// Handler of the swap chain extent
	std::vector<VkImageView> swapChainImageViews;	// Array to handle all the views into the images of the swap chain
	std::vector<VkFramebuffer> swapChainFramebuffers;	// Array of all the frame buffers of the swap chain
	
	std::vector<VkFramebuffer> swapChainFramebuffersGUI;	// Array of all the frame buffers of the swap chain

	VkFramebuffer shadowFrameBuffer;	// frame buffer to create the shadow map

	VkRenderPass renderPass;	// handler of the render pass
	VkRenderPass shadowRenderPass;	// handler of the render pass to create the shadow map
	VkRenderPass renderPassGUI; // handler of the render pass to create the GUI
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;	// handler of the pipeline layout
	VkPipeline graphicsPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout pipelineLayoutShadow;	// handler of the pipeline layout
	VkPipeline graphicsPipelineShadow;	// handler of the whole graphics pipeline

	VkPipelineLayout pipelineLayoutQuad;	// handler of the pipeline layout
	VkPipeline graphicsPipelineQuad;	// handler of the whole graphics pipeline

	VkCommandPool commandPool;	// Manages the memory that will store the buffers and command buffers
	VkCommandPool commandPoolGUI;	// Manages the memory that will store the buffers and command buffers

	VkImage depthImage;	
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	VkImage shadowImage;
	VkDeviceMemory shadowImageMemory;
	VkImageView shadowImageView;

	VkSampler textureSampler;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;


	std::vector<Vertex> verticesQuad;
	std::vector<uint32_t> indicesQuad;
	VkBuffer vertexBufferQuad;
	VkDeviceMemory vertexBufferMemoryQuad;
	VkBuffer indexBufferQuad;
	VkDeviceMemory indexBufferMemoryQuad;


	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;

	std::vector<VkBuffer> lightUniformBuffers;
	std::vector<VkDeviceMemory> lightUniformBuffersMemory;

	std::vector<VkBuffer> shadowUniformBuffers;
	std::vector<VkDeviceMemory> shadowUniformBuffersMemory;

	VkDescriptorPool descriptorPool;
	VkDescriptorPool descriptorPoolGUI;
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkCommandBuffer> commandBuffers;	// Array of a command buffer per image in the swap chain
	std::vector<VkCommandBuffer> commandBuffersGUI;	// Array of a command buffer per image in the swap chain
	std::vector<VkSemaphore> imageAvailableSemaphores;	// Array of semaphores to indicate that an image has been acquire and is ready for rendering
	std::vector<VkSemaphore> renderFinishedSemaphores;	// Array of semaphores to indicate that rendering has finished and presentation can happend
	std::vector<VkFence> inFlightFences;	// Array of all the fences that will be simultaneously built
	std::vector<VkFence> imagesInFlight;	// Array of the fences that each image will have
	size_t currentFrame = 0;
	/*
	float posCameraX = 0;
	float posCameraY = 0;
	float posCameraZ = 0;
	*/
	bool framebufferResized = false;	// Tells if the framebuffer has been resized

	float depthBiasConstant = 1.25f;	
	float depthBiasSlope = 1.75f;

	std::array<int, 5> pushConstants;	// variables to control the behaviour of the GUI
	
	bool isShowShadowMap = false;
	bool isShadowMapAsTexture = false;
	bool isDepthLight = false;
	bool isStandardLight = false;
	bool isAcne = true;
	bool isPCF = true;

	// Start GLFW and creates the window
	void initWindow() {
		glfwInit();	// Starts GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);	// It tells GLFW that it needs a window but is not an OpenGL one.

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Create the actual window with the specified Width and Height, named Vulkan
																			  // and the first nullptr is to decide which monitor to use and the second one is only for OpenGl
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);	// Allows GLFW to use the method specified in case that the window is tryed to be resized
	}

	// Uses the GLFW function to make feasible to resize the window and set that it has been resized
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {	// intialize all Vulkan Objects
		createInstance();	// Create the instance which is the way to communicate between our application and the Vulkan library
		setupDebugMessenger();	// Set up the debugger to properly show messages
		createSurface();		// Creates a surface to render images into it depending of the device where it is running
		pickPhysicalDevice();	// Selects a number of graphics cards to use
		createLogicalDevice();	// Creates queues from the ones available to the system
		createSwapChain();	// Creates the swap chain implementation
		createImageViews();	// Creates the image views for the images of the swap chain
		createRenderPass();	// Specifies how many color and depth buffers and how they should be configured 
		createRenderPassGUI();
		createShadowRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();	// Creates the whole Graphics pipeline with all the shaders
		createGraphicsPipelineShadow();	// Creates the whole Graphics pipeline with all the shaders
		createGraphicsPipelineQuad();
		createCommandPool();	// Create a manager of the memory that will store the buffers and command buffers
		createDepthResources();
		createShadowResources();
		createFramebuffers();	// Actually create the FrameBuffers with all the attachments and it will hold all the images of the swap chain
		createFramebuffersGUI();
		createShadowFramebuffers();
		createTextureSampler();	// Create a sampler for the texture
		loadModel();	// Loads the obj file containing the model
		createVertexBuffer();	// Creates the vertex buffers that will hold the vertices data
		createVertexBufferQuad();
		createIndexBuffer();	// Creates an indices buffer to hold the indices of the vertices that will be used
		createIndexBufferQuad();
		createUniformBuffers();	// Creates the object uniform buffer
		createLightUniformBuffers();	// Creates the light uniform buffer
		createShadowUniformBuffers();	// Creates the light uniform buffer
		createDescriptorPool();	// Create descriptor pools for the uniform buffer
		createDescriptorPoolGUI();
		createDescriptorSets();	// Create descriptor sets for the uniform buffer
		createCommandBuffers();	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
		createSyncObjects();	// Creates all the objects required to create synchronization
	}

	void mainLoop() {	// loops untill the window is closed
		
		ImGui::CreateContext();	// Set the environment for the gui

		ImGui::StyleColorsDark();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui_ImplGlfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};	// expose values of vulkan to the gui
		init_info.Instance = instance;
		init_info.PhysicalDevice = physicalDevice;
		init_info.Device = device;
		init_info.Allocator = nullptr;
		init_info.QueueFamily = 4;
		init_info.Queue = graphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = descriptorPoolGUI;
		init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
		init_info.ImageCount = (uint32_t)swapChainImages.size();
		init_info.CheckVkResultFn = nullptr;
		ImGui_ImplVulkan_Init(&init_info, renderPassGUI);

		VkCommandBuffer command_buffer = beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
		endSingleTimeCommands(command_buffer);
	
		bool prevShadowMap = false;	// variables to control the behaviour of the gui
		bool prevShadowMapText = false;
		bool prevDepthLight = false;
		bool prevStandardLight = false;
		bool prevAcne = true;
		bool prevPCF = true;

		while (!glfwWindowShouldClose(window)) {	// Checks if the windows has closed and loops untill it does.
			glfwPollEvents();	// ask for events

			ImGui_ImplVulkan_NewFrame();	// create a new frame for the gui
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			ImGui::Checkbox("display shadowmap", &isShowShadowMap);
			ImGui::Checkbox("display shadowmap as a texture", &isShadowMapAsTexture);
			ImGui::Checkbox("depth in lighth coords for each visible", &isDepthLight);
			ImGui::Checkbox("standard lighting", &isStandardLight);
			ImGui::Checkbox("acne removed?", &isAcne);
			ImGui::Checkbox("PCF filter activated?", &isPCF);
			ImGui::Render();

			// If there is a change in the gui
			if (isShowShadowMap != prevShadowMap || isShadowMapAsTexture != prevShadowMapText || isDepthLight != prevDepthLight ||
				isStandardLight != prevStandardLight || isAcne != prevAcne || isPCF != prevPCF)
			{
				prevShadowMap = isShowShadowMap;
				prevShadowMapText = isShadowMapAsTexture;
				prevDepthLight = isDepthLight;
				prevStandardLight = isStandardLight;
				prevAcne = isAcne;
				prevPCF = isPCF;
				recreateSwapChain();
			}
			
			drawFrame();	// do all the work untill a frame is drawn
		}
		vkDeviceWaitIdle(device);	// Waits untill the logical device finish all its tasks to be able to destroy the windows later on
	}

	// Handles all the clean up related with the swap chain
	void cleanupSwapChain() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		vkDestroyImageView(device, shadowImageView, nullptr);
		vkDestroyImage(device, shadowImage, nullptr);
		vkFreeMemory(device, shadowImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {	// for every framebuffer aka. images in the swap chain
			vkDestroyFramebuffer(device, framebuffer, nullptr);	// destroys the frame buffer
		}

		for (auto framebuffer : swapChainFramebuffersGUI) {	// for every framebuffer aka. images in the swap chain
			vkDestroyFramebuffer(device, framebuffer, nullptr);	// destroys the frame buffer
		}
		vkDestroyFramebuffer(device, shadowFrameBuffer, nullptr);

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());	// Clean up the existing command buffers without destroying them
		vkFreeCommandBuffers(device, commandPoolGUI, static_cast<uint32_t>(commandBuffersGUI.size()), commandBuffersGUI.data());	// Clean up the existing command buffers without destroying them
		
		vkDestroyPipeline(device, graphicsPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, graphicsPipelineShadow, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayoutShadow, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, graphicsPipelineQuad, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayoutQuad, nullptr);	// destroys the handler for the pipeline layout
		
		vkDestroyRenderPass(device, renderPass, nullptr);	// destroys the handler for the render pass
		vkDestroyRenderPass(device, shadowRenderPass, nullptr);	// destroys the handler for the render pass
		vkDestroyRenderPass(device, renderPassGUI, nullptr);
		for (auto imageView : swapChainImageViews) {	// for every image view
			vkDestroyImageView(device, imageView, nullptr);	// destroy it 
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);	// detroys the swap chain 

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
			vkDestroyBuffer(device, lightUniformBuffers[i], nullptr);
			vkFreeMemory(device, lightUniformBuffersMemory[i], nullptr);
			vkDestroyBuffer(device, shadowUniformBuffers[i], nullptr);
			vkFreeMemory(device, shadowUniformBuffersMemory[i], nullptr);
		}

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		vkDestroyDescriptorPool(device, descriptorPoolGUI, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	}

	void cleanup() {	 // deallocate every Vulkan Object

		cleanupSwapChain();

		vkDestroySampler(device, textureSampler, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		vkDestroyBuffer(device, indexBufferQuad, nullptr);
		vkFreeMemory(device, indexBufferMemoryQuad, nullptr);
		vkDestroyBuffer(device, vertexBufferQuad, nullptr);
		vkFreeMemory(device, vertexBufferMemoryQuad, nullptr);
	

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);	// Destroy the semaphores per the amount of simultaneous frames allowed
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);	// Destroy the fence per the amount of simultaneous frames allowed
		}

		vkDestroyCommandPool(device, commandPool, nullptr);	// Destroys the command pool buffer
		vkDestroyCommandPool(device, commandPoolGUI, nullptr);	// Destroys the command pool buffer

		vkDestroyDevice(device, nullptr);	// Destroys the logical device handler and so it cleans the related objects to it as the handlers for the queues 

		if (enableValidationLayers) {	// if validations are activated, destroy the debugMessage object
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr); // GLFW doesn't give us a function to destroy the surface, so we use the ones of the original API
		vkDestroyInstance(instance, nullptr);	// destroy the instance

		glfwDestroyWindow(window);	// destroy the window

		glfwTerminate();	// tells GLFW to finish
	}

	// Restart the swap chain in the case that there is any update related with the window
	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);	// Gather the sizes of the window with the help of GLFW
		while (width == 0 || height == 0) {	// While the sizes of the windows are minimize
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();	// Wait for events that change them
		}

		vkDeviceWaitIdle(device);	// We wait until the logical device finishes all its tasks

		cleanupSwapChain();	// Clean up all the previous information of the swap chain before reconstructing it again

		// do all the configurations again
		createSwapChain();
		createImageViews();
		createRenderPass();
		createRenderPassGUI();
		createShadowRenderPass();
		createGraphicsPipeline();
		createGraphicsPipelineShadow();
		createGraphicsPipelineQuad();
		createDepthResources();
		createShadowResources();
		createFramebuffers();
		createFramebuffersGUI();
		createShadowFramebuffers();
		createUniformBuffers();
		createLightUniformBuffers();
		createShadowUniformBuffers();
		createDescriptorPool();
		createDescriptorPoolGUI();
		createDescriptorSets();
		createCommandBuffers();

		ImGui::CreateContext();	// recreate the gui

		ImGui::StyleColorsDark();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui_ImplGlfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = instance;
		init_info.PhysicalDevice = physicalDevice;
		init_info.Device = device;
		init_info.Allocator = nullptr;
		init_info.QueueFamily = 4;
		init_info.Queue = graphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = descriptorPoolGUI;
		init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
		init_info.ImageCount = (uint32_t)swapChainImages.size();
		init_info.CheckVkResultFn = nullptr;
		ImGui_ImplVulkan_Init(&init_info, renderPassGUI);

		VkCommandBuffer command_buffer = beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
		endSingleTimeCommands(command_buffer);

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	
		ImGui::Checkbox("display shadowmap", &isShowShadowMap);
		ImGui::Checkbox("display shadowmap as a texture", &isShadowMapAsTexture);
		ImGui::Checkbox("depth in lighth coords for each visible", &isDepthLight);
		ImGui::Checkbox("standard lighting", &isStandardLight);
		ImGui::Checkbox("acne removed?", &isAcne);
		ImGui::Checkbox("PCF filter activated?", &isPCF);

		ImGui::Render();

		createCommandPoolGUI(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		createCommandBuffersGUI();

	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {	// check if the validation layer are activated and if all of the desired ones are in the system.
																		// if not, throw a runtime error
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;			// type or identifier of the struct
		appInfo.pApplicationName = "Hello Triangle";				// gives the name to the application
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);		// number of the version of the created application
		appInfo.pEngineName = "No Engine";							// name of the engine that creates the application
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);			// number of the version of the engine 
		appInfo.apiVersion = VK_API_VERSION_1_0;					// is the highest version of Vulkan the application is going to be allowed to run on.

		VkInstanceCreateInfo createInfo = {};	// it contains mandatory information to create the instance
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;	// type or identifier of the struct
		createInfo.pApplicationInfo = &appInfo;	// helps implementations recognize behavior inherent to applications

		auto extensions = getRequiredExtensions(); // call to the private method that controls which extensions are going to be used
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());	// number of global extensions enabled
		createInfo.ppEnabledExtensionNames = extensions.data();	// contains the names of the extensions enabled

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		if (enableValidationLayers) {	// if the validation layers check is activated
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());	// number of global layers enabled
			createInfo.ppEnabledLayerNames = validationLayers.data();	// contains the names of the layers enabled

			populateDebugMessengerCreateInfo(debugCreateInfo); // create the debugger with the information in createInfo	
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;	// pointer to define the extension of the original structure
		}
		else {
			createInfo.enabledLayerCount = 0;	// set the number of global layers to 0

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {	// Checks if the creation of the instance has been successful or not
			throw std::runtime_error("failed to create instance!");
		}
	}

	// Abstraction to allow the creation of two different debuggers for vkCreateInstance and vkDestroyInstance
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;	// type or identifier of the struct
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;	// Specify level of severity
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; // notify the contents of the callback
		createInfo.pfnUserCallback = debugCallback;	// pointer to the callback function 
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;	// if the validation layer is not activated then don't do anything

		VkDebugUtilsMessengerCreateInfoEXT createInfo;	// create a struct to create the messenger
		populateDebugMessengerCreateInfo(createInfo);	// create the debugger with the information in createInfo

		// function to create the VkDebugUtilsMessengerEXT object which returns if it was a success or not
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");	// throw and error
		}
	}

	// Creates a surface to render images into it depending of the device where it is running
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {	// uses the GLFW library to create a window surface for us
			throw std::runtime_error("failed to create window surface!");
		}
	}

	// Selects which graphics cards are going to be used
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);	// query for the number of graphics card available

		if (deviceCount == 0) {	// if there is not any graphics card that supports vulkan throw a runtime error.
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);	// array that will hold the details of every graphics card available to the system
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // fill in the array with the contents of each graphic cards that supports Vulkan

		for (const auto& device : devices) {	// check for every available graphics card
			if (isDeviceSuitable(device)) {		// if it is suitable for our requirements if so select it and stop serching
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {	// If any of them was suitable throw a runtime error
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// Creates the logical device selecting which of the available queues is going to be used
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);	// retrieve the indices of the available queues

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };	// get the values of the different queues available

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {	// loop through the queues available and set details for each of them
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;	// type or identifier of the struct
			queueCreateInfo.queueFamilyIndex = queueFamily;	// set the value of this particular queue to be created on the device
			queueCreateInfo.queueCount = 1;	// specify the number of queues to create
			queueCreateInfo.pQueuePriorities = &queuePriority;	// pointer to specify properties that apply to each created queue
			queueCreateInfos.push_back(queueCreateInfo);	// set the values configured by each queue
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;	// type or identifier of the struct

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());	// specify the size of the queue
		createInfo.pQueueCreateInfos = queueCreateInfos.data();	// set the data of the queue

		createInfo.pEnabledFeatures = &deviceFeatures;	// pointer to enable the configuration made from the struct

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()); //	number of device extensions to enable
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();	// data of the extensions enabled

		if (enableValidationLayers) {	// if the validation layers are activated. This is not really need it for lastest versions of Vulkan, 
										// because now there are not distinctions between instance and device validation layers
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());	// set the number of validation layers enabled
			createInfo.ppEnabledLayerNames = validationLayers.data();	// set the data of those layers that are enabled
		}
		else {
			createInfo.enabledLayerCount = 0;	// otherwise there are not any layers
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {	// instantiate the logical device with the data of the structs and throw an error if it fails
			throw std::runtime_error("failed to create logical device!");
		}

		// the input of this functions are logical device, queue family, queue index and the pointer to return the handler 
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);	// retrieve queue handle for the graphics queue
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);		// retrieve queue handle for the present queue
	}

	// Creates the whole swap chain which are the way in which the images are shown in the screen and how it synchronize with the refresh rate of the screen
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);	// retrieves the neccessary information to create the swap chain

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);	// select a surface format
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);	// select a present mode
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);	// select a swap extent

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;	// specify the number of images it needs to function, to avoid driver waiting we use 1 more
																				// than the minimum time
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {	// Check that we don't exceed the maximum number of images
																															// Where 0 in this case means there is not a maximum number of images
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;	// type of the structure
		createInfo.surface = surface;	// surface in which the swap chain will present images

		createInfo.minImageCount = imageCount;	// min number of images that the application needs
		createInfo.imageFormat = surfaceFormat.format;	// set the format of the swap chain
		createInfo.imageColorSpace = surfaceFormat.colorSpace;	// set the color space
		createInfo.imageExtent = extent;	// set the image extent
		createInfo.imageArrayLayers = 1;	// is the number of views in a multiview/stereo surface
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;	// describe the intended usage of the images in the swap chain

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);	// query the indices of the queues in the system
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };	// array of the two different queues we have

		if (indices.graphicsFamily != indices.presentFamily) {	// if they have different queue families we will have to use concurrent mode
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;	// We set up that there will be 2 concurrent queues
			createInfo.pQueueFamilyIndices = queueFamilyIndices; // we give the indices of the queues
		}
		else {	// if the share family queue we can use exclusive mode where only one queue has acces at the same time to an image which usually improves performance
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;	// We don't want to set any transformations to the images in the swap chain so we let them as they are
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;	// specify the desired alpha channel for blending
		createInfo.presentMode = presentMode;	// set the present mode
		createInfo.clipped = VK_TRUE;	// we allow the system to remove anything that has been clipped

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {	// We finally create the swap chain with the information of the struct and throw a
																							// runtime error if it fails
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);	// get the number of images
		swapChainImages.resize(imageCount);	// resize the number of images in the struct to the actual number of images
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());	// fill in the data of the images in the array

		swapChainImageFormat = surfaceFormat.format;	// set the format for the member variable
		swapChainExtent = extent;	// set the format for the member variable
	}

	// Creates an image view for every image of the swap chain
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	// Create a new render pass for the gui
	void createRenderPassGUI() {
		VkAttachmentDescription attachment = {};	// single color buffer 
		attachment.format = swapChainImageFormat;	// we use the same format as in the swap chain
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;	// we don't do multisampling right now so, we have only 1 sample
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;	// The data in the attachment before rendering will be set to a constant value
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	// The data in the attachment after rendering in this case will be stored and can later on be readed from the buffer
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	// As we don't use the stencil buffer we have undefined contents in that buffer
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	// The same with this one, after rendering the contents will be undefined as we do not worry about them
		attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies how should be the layout before the render pass, in this case is undefined
		attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;	// specifies the layout to automatically transition in this case by presenting the image to the swap chain

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;	// index of the reference attachment
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the desired layout during supasses

		// A subpass is a subsequent rendering pass using information of previous subpasses 
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;	// Specifies the type of subpass that we will do in our case graphics
		subpass.colorAttachmentCount = 1;	// is the number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef;	// pointer to the color attachment ref that will be used in that subpass

		// The transition between subpasses are controlled by the dependency and we will have to configure when should the start transition should start
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;	// is the index of the first subpass in the dependency
		dependency.dstSubpass = 0;	// is the index of the second subpass in the dependency
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies when should the first subpass should start (in which stage)
		dependency.srcAccessMask = 0;	// Specifies the source access 
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies until which stage it should wait 
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;	// Specifies the destination access mask

		std::array<VkAttachmentDescription, 1> attachments = { attachment };
		// Has all the data to be able to construct the render pass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// number of attachments in this render pass
		renderPassInfo.pAttachments = attachments.data();	// pointer to the color attachment buffer
		renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
		renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
		renderPassInfo.dependencyCount = 1;	// is the number of memory dependencies between subpasses
		renderPassInfo.pDependencies = &dependency;	// pointer to the dependencies struct

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPassGUI) != VK_SUCCESS) {	// Creates the render pass with the data of the struct and 
																								// throw a runtime error if it fails
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Configures how to configurate the color buffer and how could the render pass and subpasses be implemented
	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};	// single color buffer 
		colorAttachment.format = swapChainImageFormat;	// we use the same format as in the swap chain
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;	// we don't do multisampling right now so, we have only 1 sample
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;	// The data in the attachment before rendering will be set to a constant value
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	// The data in the attachment after rendering in this case will be stored and can later on be readed from the buffer
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	// As we don't use the stencil buffer we have undefined contents in that buffer
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	// The same with this one, after rendering the contents will be undefined as we do not worry about them
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;	// specifies how should be the layout before the render pass, in this case is undefined
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the layout to automatically transition in this case by presenting the image to the swap chain
		

		VkAttachmentDescription depthAttachment = {};	// Set up all the information for the depth buffer to be able to work
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;	// index of the reference attachment
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the desired layout during supasses

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// A subpass is a subsequent rendering pass using information of previous subpasses 
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;	// Specifies the type of subpass that we will do in our case graphics
		subpass.colorAttachmentCount = 1;	// is the number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef;	// pointer to the color attachment ref that will be used in that subpass
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		// The transition between subpasses are controlled by the dependency and we will have to configure when should the start transition should start
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;	// is the index of the first subpass in the dependency
		dependency.dstSubpass = 0;	// is the index of the second subpass in the dependency
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies when should the first subpass should start (in which stage)
		dependency.srcAccessMask = 0;	// Specifies the source access 
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies until which stage it should wait 
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;	// Specifies the destination access mask

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		// Has all the data to be able to construct the render pass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// number of attachments in this render pass
		renderPassInfo.pAttachments = attachments.data();	// pointer to the color attachment buffer
		renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
		renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
		renderPassInfo.dependencyCount = 1;	// is the number of memory dependencies between subpasses
		renderPassInfo.pDependencies = &dependency;	// pointer to the dependencies struct

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {	// Creates the render pass with the data of the struct and 
																								// throw a runtime error if it fails
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Configures how to configurate the color buffer and how could the render pass and subpasses be implemented for the shadow map creation
	void createShadowRenderPass() {
		VkAttachmentDescription depthAttachment = {};	// Set up all the information for the depth buffer to be able to work
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL; // VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 0;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// A subpass is a subsequent rendering pass using information of previous subpasses 
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;	// Specifies the type of subpass that we will do in our case graphics
		subpass.colorAttachmentCount = 0;	// is the number of color attachments
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		// The transition between subpasses are controlled by the dependency and we will have to configure when should the start transition should start
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;	// controls how is going to be implicitly synchronize the information between render passes.
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Has all the data to be able to construct the render pass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
		renderPassInfo.attachmentCount = 1;	// number of attachments in this render pass
		renderPassInfo.pAttachments = &depthAttachment;	// pointer to the color attachment buffer
		renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
		renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());	// is the number of memory dependencies between subpasses
		renderPassInfo.pDependencies = dependencies.data();	// pointer to the dependencies struct

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowRenderPass) != VK_SUCCESS) {	// Creates the render pass with the data of the struct and 
																								// throw a runtime error if it fails
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Describe the layout that each of the uniform buffers will have
	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding lboLayoutBinding = {};
		lboLayoutBinding.binding = 1;
		lboLayoutBinding.descriptorCount = 1;
		lboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		lboLayoutBinding.pImmutableSamplers = nullptr;
		lboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding shadowLayoutBinding = {};	// this descriptors allows the shaders to access a uniform resource
		shadowLayoutBinding.binding = 2;
		shadowLayoutBinding.descriptorCount = 1;
		shadowLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		shadowLayoutBinding.pImmutableSamplers = nullptr;
		shadowLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding shadowDepthLayoutBinding = {};	// this descriptors allows the shaders to access an image resource
		shadowDepthLayoutBinding.binding = 3;
		shadowDepthLayoutBinding.descriptorCount = 1;
		shadowDepthLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		shadowDepthLayoutBinding.pImmutableSamplers = nullptr;
		shadowDepthLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 4> bindings = { uboLayoutBinding, lboLayoutBinding, shadowLayoutBinding, shadowDepthLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shaders/vert.spv");	// Read the already compiled vertex shader file
		auto fragShaderCode = readFile("shaders/frag.spv"); // Read the already compiled fragment shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);	// Wrap the bytecode of the fragment shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;	// specify which stage of the pipeline is defining
		fragShaderStageInfo.module = fragShaderModule;	// Contains the shader object
		fragShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_FALSE;	// enables or disable applying a depth bias 

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;	// specifies the number of samples used in rasterization

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// proccess the combination of colors from the fragment shader and the color that was previously in the buffer
		// it does it by attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		// specifies which channels are open to be modified bitwise
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;	// if it's set to false it will just use the color out of the fragments shader otherwise it will perform color blending
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; 
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.logicOpEnable = VK_FALSE;	// Controls wether to apply Logical Operations
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	// Selects the logical operation to apply
		colorBlending.attachmentCount = 1;	// Defines the number of VkPipelineColorBlendAttachmentState objects 
		colorBlending.pAttachments = &colorBlendAttachment;	// pointer to the attachments
		colorBlending.blendConstants[0] = 0.0f;	// constant values that are use while blending for R, G, B, A
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstants);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 2;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayout;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = renderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);	// Destroy the fragment shader module
		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	// create the pipeline to create a quad wiht the shadow map to be visible
	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipelineQuad() {
		auto vertShaderCode = readFile("shaders/vertQuad.spv");	// Read the already compiled vertex shader file
		auto fragShaderCode = readFile("shaders/fragQuad.spv"); // Read the already compiled fragment shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);	// Wrap the bytecode of the fragment shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;	// specify which stage of the pipeline is defining
		fragShaderStageInfo.module = fragShaderModule;	// Contains the shader object
		fragShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_FALSE;	// enables or disable applying a depth bias 

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;	// specifies the number of samples used in rasterization

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// proccess the combination of colors from the fragment shader and the color that was previously in the buffer
		// it does it by attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		// specifies which channels are open to be modified bitwise
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;	// if it's set to false it will just use the color out of the fragments shader otherwise it will perform color blending
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.logicOpEnable = VK_FALSE;	// Controls wether to apply Logical Operations
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	// Selects the logical operation to apply
		colorBlending.attachmentCount = 1;	// Defines the number of VkPipelineColorBlendAttachmentState objects 
		colorBlending.pAttachments = &colorBlendAttachment;	// pointer to the attachments
		colorBlending.blendConstants[0] = 0.0f;	// constant values that are use while blending for R, G, B, A
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutQuad) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 2;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayoutQuad;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = renderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipelineQuad) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);	// Destroy the fragment shader module
		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	// Pipeline to create the shadow map writting the depth to it
	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipelineShadow() {
		auto vertShaderCode = readFile("shaders/shadowVert.spv");	// Read the already compiled vertex shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_TRUE;	// enables or disable applying a depth bias 

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;	// specifies the number of samples used in rasterization

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.attachmentCount = 0;	// Defines the number of VkPipelineColorBlendAttachmentState objects 


		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutShadow) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 1;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayoutShadow;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = shadowRenderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipelineShadow) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	// Create a new frame buffer for the gui
	// Create the frame buffer that will hold all of the images of the swap chain and will wrap up all of the others buffers
	void createFramebuffersGUI() {
		swapChainFramebuffersGUI.resize(swapChainImageViews.size());	// Resize the buffer to be able to contain all the images of the swap chain

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {	// iterate through every image of the swap chain and create a buffer for each
			std::array<VkImageView, 1> attachments = {
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
			framebufferInfo.renderPass = renderPassGUI;	// indicates which renderPass will be used and how it is configured 
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// We will only use one attachments to the buffer that is the color buffer
			framebufferInfo.pAttachments = attachments.data();	// pointer to the attachments to use in the buffer
			framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffersGUI[i]) != VK_SUCCESS) {	// Create the FrameBuffer object using the data of the struct
																													// Throw a runtime error exception if something fails
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}


	// Create the frame buffer that will hold all of the images of the swap chain and will wrap up all of the others buffers
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());	// Resize the buffer to be able to contain all the images of the swap chain

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {	// iterate through every image of the swap chain and create a buffer for each
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
			framebufferInfo.renderPass = renderPass;	// indicates which renderPass will be used and how it is configured 
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// We will only use one attachments to the buffer that is the color buffer
			framebufferInfo.pAttachments = attachments.data();	// pointer to the attachments to use in the buffer
			framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {	// Create the FrameBuffer object using the data of the struct
																													// Throw a runtime error exception if something fails
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	// Create a new shadow buffer for the shadow map with only a depht attachment
	// Create the frame buffer that will hold all of the images of the swap chain and will wrap up all of the others buffers
	void createShadowFramebuffers() {
		
		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
		framebufferInfo.renderPass = shadowRenderPass;	// indicates which renderPass will be used and how it is configured 
		framebufferInfo.attachmentCount = 1;	// We will only use one attachments to the buffer that is the color buffer
		framebufferInfo.pAttachments = &shadowImageView;	// pointer to the attachments to use in the buffer
		framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadowFrameBuffer) != VK_SUCCESS) {	// Create the FrameBuffer object using the data of the struct
																												// Throw a runtime error exception if something fails
			throw std::runtime_error("failed to create framebuffer!");
		}
		
	}


	// Create a manager of the memory that will store the buffers and command buffers
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);	// first we need to find the index of the different queues available

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;	// set the type of the struct
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();	// as we want to use the draw call, we will need the graphics queue

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {	// Create the command pool using the data of the struct 
																							// Throw a runtime error if something fails
			throw std::runtime_error("failed to create command pool!");
		}
	}

	// Create all the info required for the depth buffer to work
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();	// search for an adequate format

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);	// Create an image view that will be used for the depth
	}

	// Create all the info required for the image that will have the shadow map
	void createShadowResources() {
		VkFormat depthFormat = findDepthFormat();	// search for an adequate format

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowImage, shadowImageMemory);
		shadowImageView = createImageView(shadowImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);	// Create an image view that will be used for the depth
	}

	// find the adequate format for the depth buffer to work
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}
	// Calls a method with the hardcoded parameters to find a suitable format
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	
	// Creates a sampler for the texture
	void createTextureSampler() {
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;	// 1.0f
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	// Create an image view depending on the type of flag
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	// Creates an image with the data of the parameters. Also check for the memory requirements and allocate space for that image
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format; 
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	// Execute commands buffers
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	// Perform a copy of the information in the buffer to the image
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	// fill in the array of vertices and the array of indices using an Obj file
	void loadModel() {		
		std::ifstream inFile;

		inFile.open("viking_room.obj");
		if (!inFile) {
			throw std::runtime_error("Error opening Model object");
		}

		std::vector<glm::vec3> tempCoordsVertex;	// stores all the position values of the vertices
		std::vector<glm::vec3> tempNormalCoordsVertex;	// stores all the normal values of the vertices
		std::vector<glm::vec2> tempTextCoordsVertex;	// stores all the texture coordinates of the vertices
		std::string line;
		double x, y, z;
		std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

		while (!inFile.eof()) {
			std::getline(inFile, line);

			if (line[0] == 'v' && line[1] == ' ')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				//pos = line.find(' ');
				//line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				//tempCoordsVertex.push_back({ x,y,z });
				Vertex v = {};
				v.pos = glm::vec3(x, y, z);
				vertices.push_back(v);
			}
			else if (line[0] == 'v' && line[1] == 'n')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				//pos = line.find(' ');
				//line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				tempNormalCoordsVertex.push_back({ x,y,z });
			}
			else if (line[0] == 'v' && line[1] == 't')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				//pos = line.find(' ');
				//line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				tempTextCoordsVertex.push_back({ x, 1.0f - y });	// change it so that 0 value is top left corner
			}
			else if (line[0] == 'f')	// read every face and triangulate them
			{
				// The values are set to -1 because the Obj format indices starts in 1 and the one in the arrays starts in 0
				long posIndex1 = -1;	// hold the position index of the vertex 1
				long textIndex1 = -1;	// hold the texture index of the vertex 1
				long normalIndex1 = -1;	// hold the normal index of the vertex 1

				long posIndex2 = -1;
				long textIndex2 = -1;
				long normalIndex2 = -1;

				long posIndex3 = -1;
				long textIndex3 = -1;
				long normalIndex3 = -1;

				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find('/');
				posIndex1 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				textIndex1 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				normalIndex1 += std::stol(line.substr(0, pos));

				// Second
				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find('/');
				posIndex2 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				textIndex2 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				normalIndex2 += std::stol(line.substr(0, pos));

				// Third
				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find('/');
				posIndex3 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				textIndex3 += std::stol(line.substr(0, pos));

				pos = line.find('/');
				line.erase(0, pos + 1);
				pos = line.find('/');
				normalIndex3 += std::stol(line.substr(0, pos));

				vertices[posIndex1].color = glm::vec3(1.0f, 1.0f, 1.0f);
				vertices[posIndex1].texCoord = tempTextCoordsVertex[textIndex1];
				vertices[posIndex1].normalCoord = tempNormalCoordsVertex[normalIndex1];

				vertices[posIndex2].color = glm::vec3(1.0f, 1.0f, 1.0f);
				vertices[posIndex2].texCoord = tempTextCoordsVertex[textIndex2];
				vertices[posIndex2].normalCoord = tempNormalCoordsVertex[normalIndex2];

				vertices[posIndex3].color = glm::vec3(1.0f, 1.0f, 1.0f);
				vertices[posIndex3].texCoord = tempTextCoordsVertex[textIndex3];
				vertices[posIndex3].normalCoord = tempNormalCoordsVertex[normalIndex3];

				indices.push_back(posIndex1);
				indices.push_back(posIndex2);
				indices.push_back(posIndex3);

			}
		}
	
		indicesQuad.push_back(0);	// load the indices for the quad
		indicesQuad.push_back(1);
		indicesQuad.push_back(2);
		indicesQuad.push_back(1);
		indicesQuad.push_back(3);
		indicesQuad.push_back(2);

		// Create the 4 vertices that will create the quad
		Vertex vertex0 = {};	
		vertex0.pos = glm::vec3(-1.15f, -1.05f, 0.0f);
		vertex0.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex0.texCoord = glm::vec2(0.0f, 0.0f);
		vertex0.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);

		Vertex vertex1 = {};	
		vertex1.pos = glm::vec3(-1.15f,1.25f, 0.0f);
		vertex1.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex1.texCoord = glm::vec2(0.0f, 1.0f);
		vertex1.normalCoord = glm::vec3(0.0f,0.0f, 1.0f);

		Vertex vertex2 = {};	
		vertex2.pos = glm::vec3(1.15f, -1.05f, 0.0f);
		vertex2.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex2.texCoord = glm::vec2(1.0f, 0.0f);
		vertex2.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);

		Vertex vertex3 = {};	
		vertex3.pos = glm::vec3(1.15f, 1.25f, 0.0f);
		vertex3.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex3.texCoord = glm::vec2(1.0f, 1.0f);
		vertex3.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);
		
		// push those vertices
		verticesQuad.push_back(vertex0);
		verticesQuad.push_back(vertex1);
		verticesQuad.push_back(vertex2);
		verticesQuad.push_back(vertex3);

		inFile.close();
	}

	// Creates a buffer to hold the information of the vertices, allocating memory and using a staging buffer
	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the information of the vertices of the quad
	void createVertexBufferQuad() {
		VkDeviceSize bufferSize = sizeof(verticesQuad[0]) * verticesQuad.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, verticesQuad.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferQuad, vertexBufferMemoryQuad);

		copyBuffer(stagingBuffer, vertexBufferQuad, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a buffer to hold the indices of the vertices, allocating memory and using a staging buffer
	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the indices of the vertices of the quad
	void createIndexBufferQuad() {
		VkDeviceSize bufferSize = sizeof(indicesQuad[0]) * indicesQuad.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indicesQuad.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBufferQuad, indexBufferMemoryQuad);

		copyBuffer(stagingBuffer, indexBufferQuad, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Create the uniform buffer for the object one per image in the swap chain
	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(swapChainImages.size());
		uniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
		}
	}

	// Create the uniform buffer for the light one per image in the swap chain
	void createLightUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferLight);

		lightUniformBuffers.resize(swapChainImages.size());
		lightUniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightUniformBuffers[i], lightUniformBuffersMemory[i]);
		}
	}

	// Create the uniform buffer for the shadow one per image in the swap chain
	void createShadowUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferShadow);

		shadowUniformBuffers.resize(swapChainImages.size());
		shadowUniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, shadowUniformBuffers[i], shadowUniformBuffersMemory[i]);
		}
	}



	// Creates a descriptor pool holding information for the Gui
	void createDescriptorPoolGUI() {
		std::array<VkDescriptorPoolSize, 11> poolSizes = {};
		poolSizes[0] = { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 };
		poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 };
		poolSizes[2] = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 };
		poolSizes[3] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 };
		poolSizes[4] = { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 };
		poolSizes[5] = { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 };
		poolSizes[6] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 };
		poolSizes[7] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 };
		poolSizes[8] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 };
		poolSizes[9] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 };
		poolSizes[10] = { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 };
		

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPoolGUI) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	// Creates a descriptor pool holding all the information of the buffers that need to be transfered to the GPU
	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 4> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[3].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	// Holds information of the different descriptors that will be used
	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorBufferInfo bufferLightInfo = {};
			bufferLightInfo.buffer = lightUniformBuffers[i];
			bufferLightInfo.offset = 0;
			bufferLightInfo.range = sizeof(UniformBufferLight);

			VkDescriptorBufferInfo bufferShadowInfo = {};
			bufferShadowInfo.buffer = shadowUniformBuffers[i];
			bufferShadowInfo.offset = 0;
			bufferShadowInfo.range = sizeof(UniformBufferShadow);

			VkDescriptorImageInfo shadowImageInfo = {};		
			shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;	
			shadowImageInfo.imageView = shadowImageView;
			shadowImageInfo.sampler = textureSampler;


			std::array<VkWriteDescriptorSet, 4> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &bufferLightInfo;

			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = descriptorSets[i];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pBufferInfo = &bufferShadowInfo;

			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = descriptorSets[i];
			descriptorWrites[3].dstBinding = 3;
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].pImageInfo = &shadowImageInfo;	// tell vulkan that for binding 4 will have the shadow map texture
		
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	// General method to create any kind of buffer allocating memory and binding it
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	// Auxiliar method for executing command buffers to be called at the beginning of the execution
	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	// Auxiliar method for executing command buffers to be called at the end of the execution
	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	// Performs the copy of a source buffer in a destination buffer
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
	void createCommandBuffers() {
		commandBuffers.resize(swapChainFramebuffers.size());	// resize the command buffer's array to be equal to the size of the swap chain images

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;	// Set the type of the struct
		allocInfo.commandPool = commandPool;	// pointer to the command pool from which the command buffers will be allocated from
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;	// primary level allows to be submitted to a queue for execution but it cannot be called from other command's buffers
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();	// Is the number of command buffers and will be equal to the number of images in the swap chain

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {	// Create the command buffer using the data from the struct
																									// Throw a runtime error if something fails
			throw std::runtime_error("failed to allocate command buffers!");
		}


		VkCommandBufferBeginInfo beginInfo = {};	// specifies the initial state of the command buffer
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;	// Set the type of the struct
		for (size_t i = 0; i < commandBuffers.size(); i++) {	// For every command buffer 

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {	// create the begin command buffer with the data of the struct
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			
			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };

			// First pass where we create the shadow map
			{
				// Drawing will start by performing a render pass
				VkRenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;	// Set the type of the struct
				renderPassInfo.renderPass = shadowRenderPass;	// pointer to the struct that holds the configuration of the shadow render pass
				renderPassInfo.framebuffer = shadowFrameBuffer;	// specify which frame buffer will be used for the render pass in our case it will be the attachment of the depth buffer
				renderPassInfo.renderArea.offset = { 0, 0 };	// We don't want any offset to the render area because we want it to fit with the one of the attachments
				renderPassInfo.renderArea.extent = swapChainExtent;	// Define the area as the one defined in the swap chain

				std::array<VkClearValue, 1> clearValues = {};
				clearValues[0].depthStencil = { 1.0f, 0 };

				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				// records a command in the specified command buffer as the first parameter, the second parameter will be the data of the struct we've just created,
				// The third argument specifies that the render pass commands will only be executed from this command buffer
				vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
				vkCmdSetDepthBias(commandBuffers[i], depthBiasConstant, 0.0f, depthBiasSlope);	
		
				// bind the vertex and index buffers
				vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);	// Binds the vertex buffer to be used for the draw call
				vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

				// bind the shadow pipeline
				vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineShadow);
				vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutShadow, 0, 1, &descriptorSets[i], 0, nullptr);

				vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);	

				vkCmdEndRenderPass(commandBuffers[i]);
			}
			

			// Second pass where we draw the scene with the shadow map or the shadow map itself over the quad
			VkBuffer vertexBuffersQuad[] = { vertexBufferQuad };
			{
				// Drawing will start by performing a render pass
				VkRenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;	// Set the type of the struct
				renderPassInfo.renderPass = renderPass;	// pointer to the struct that holds the configuration of a render pass
				renderPassInfo.framebuffer = swapChainFramebuffers[i];	// specify which frame buffer will be used for the render pass in our case it will be the attachment of the color buffer
				renderPassInfo.renderArea.offset = { 0, 0 };	// We don't want any offset to the render area because we want it to fit with the one of the attachments
				renderPassInfo.renderArea.extent = swapChainExtent;	// Define the area as the one defined in the swap chain

				std::array<VkClearValue, 2> clearValues = {};
				clearValues[0].color = { 1.0f, 0.8f, 1.0f, 1.0f };	// It will define the color of the background
				clearValues[1].depthStencil = { 1.0f, 0 };

				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				// records a command in the specified command buffer as the first parameter, the second parameter will be the data of the struct we've just created,
				// The third argument specifies that the render pass commands will only be executed from this command buffer
				vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
				

				// Shadow map display
				if (isShowShadowMap) {
					vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersQuad, offsets);
					vkCmdBindIndexBuffer(commandBuffers[i], indexBufferQuad, 0, VK_INDEX_TYPE_UINT32);
					vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineQuad);
					vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutQuad, 0, 1, &descriptorSets[i], 0, NULL);
					vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indicesQuad.size()), 1, 0, 0, 0);
				}
				else
				{
					vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);	// Binds the vertex buffer to be used for the draw call
					vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

					// send which behaviour will have the shader depending on the gui
					pushConstants[0] = isShadowMapAsTexture == true ? 1 : 0;
					pushConstants[1] = isDepthLight == true ? 1 : 0;
					pushConstants[2] = isStandardLight == true ? 1 : 0;
					pushConstants[3] = isAcne == true ? 1 : 0;
					pushConstants[4] = isPCF == true ? 1 : 0;

					vkCmdPushConstants(
						commandBuffers[i],
						pipelineLayout,
						VK_SHADER_STAGE_FRAGMENT_BIT,
						0,
						sizeof(pushConstants),
						pushConstants.data());
					
					// The first parameter to bind the pipeline is the command buffer that we are configuring, the second one will be if the pipeline is a graphics one or a compute.
					// The third parameter is the data from the struct when we configured the graphics pipeline
					vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
					vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

					// Call to finally draw the triangle specifiying first the command buffer,
					// Second is the number of vertex that will be draw
					// Third is for instance rendering, as we are not using it we use 1
					// Fourth defines the lowest vertex that we have to have it as a first vertex
					// Fifth defines the lowest value for instance rendering
					vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
				}

				vkCmdEndRenderPass(commandBuffers[i]);
			}

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {	// Create the command buffer using the data from the struct 
																		// Throw a runtime error if something fails
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	// Creates all the objects required to create synchronization
	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);	// resize the number of semaphores to the number of simultaneous frames
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);	// resize in the same way for the fences
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);	// resize to have the number of fences equal to the number of images in the swap chain

		// Struct to give the info of the sempahores
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;	// Set the type of the struct

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;	// Set the type of the struct
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;	// the fence starts being signaled

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||		// Create the semaphores using the data from the structs
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {	// Creates the fence using the data from the struct
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	// Updates the information related to the uniform buffer inside of the drawFrame loop
	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};	// fill in all the information required to the struct
		glm::vec3 translation(0.0f, -0.5f, -5.0f);
		glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
		ubo.model = glm::scale(model, { 1.3, 1.3, 1.3 });
		ubo.model = glm::rotate(ubo.model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.model = glm::rotate(ubo.model, (time/10) * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));	// apply some rotation to be different points of view
		
		ubo.view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
	}

	// Updates the information related to the uniform buffer inside of the drawFrame loop
	void updateLightUniformBuffer(uint32_t currentImage) {
		std::ifstream inFile;  // Input file opening and checking

		inFile.open("VikingRoomMaterial.mtl");
		if (!inFile) {
			throw std::runtime_error("Error opening Model object");
		}

		float specularHighlight;
		glm::vec3 ambientColor;
		glm::vec3 diffuseColor;
		glm::vec3 specularColor;
		glm::vec3 emissiveColor;

		std::string line;
		double x, y, z;

		while (!inFile.eof()) {		// Read the .mlt file with the information of the material
			std::getline(inFile, line);

			if (line[0] == '\t' && line[1] == 'N' && line[2] == 's')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				specularHighlight = std::stof(line.substr(0, pos));
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'a')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				ambientColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'd')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				diffuseColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 's')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				specularColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'e')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				emissiveColor = { x, y, z };
			}
		}
		inFile.close();
		
		UniformBufferLight ubl = {};	// After reading the file we can fill in all the information that we will use in this UniformBuffer 
		ubl.specularHighlight = specularHighlight;
		ubl.ambientColor = ambientColor;
		ubl.diffuseColor = diffuseColor;
		ubl.specularColor = specularColor;
		ubl.emissiveColor = emissiveColor;
		ubl.lightPosition = glm::vec4(0.1f, 0.0f, -2.0f, 1.0f);

		void* data;
		vkMapMemory(device, lightUniformBuffersMemory[currentImage], 0, sizeof(ubl), 0, &data);
		memcpy(data, &ubl, sizeof(ubl));
		vkUnmapMemory(device, lightUniformBuffersMemory[currentImage]);
	}

	void updateShadowUniformBuffer(uint32_t currentImage) {

		UniformBufferShadow ubs = {};	// fill in all the information required to the struct
		ubs.view = glm::lookAt(glm::vec3(0.1f, 0.0f, -2.0f), glm::vec3(0.0f, -0.5f, -5.0f), glm::vec3(0.0f, 1.0f, 0.0f));	// set the coordinates of the lights view
		ubs.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubs.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, shadowUniformBuffersMemory[currentImage], 0, sizeof(ubs), 0, &data);
		memcpy(data, &ubs, sizeof(ubs));
		vkUnmapMemory(device, shadowUniformBuffersMemory[currentImage]);
	}


	void createCommandBuffersGUI() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;	// Set the type of the struct
		allocInfo.commandPool = commandPoolGUI;	// pointer to the command pool from which the command buffers will be allocated from
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;	// primary level allows to be submitted to a queue for execution but it cannot be called from other command's buffers
		allocInfo.commandBufferCount = (uint32_t)commandBuffersGUI.size();	// Is the number of command buffers and will be equal to the number of images in the swap chain
		
		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffersGUI.data()) != VK_SUCCESS) {	// Create the command buffer using the data from the struct
																									// Throw a runtime error if something fails
			throw std::runtime_error("failed to allocate command buffers!");
		}

		VkCommandBufferBeginInfo beginInfo = {};	// specifies the initial state of the command buffer
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;	// Set the type of the struct
		for (size_t i = 0; i < commandBuffersGUI.size(); i++)	// For every command buffer 
		{
			if (vkBeginCommandBuffer(commandBuffersGUI[i], &beginInfo) != VK_SUCCESS) {	// create the begin command buffer with the data of the struct
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			
			// Render pass to create the gui with the updated values of that instance
			{
				VkRenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;	// Set the type of the struct
				renderPassInfo.renderPass = renderPassGUI;	// pointer to the struct that holds the configuration of a render pass
				renderPassInfo.framebuffer = swapChainFramebuffersGUI[i];	// specify which frame buffer will be used for the render pass in our case it will be the attachment of the color buffer
				renderPassInfo.renderArea.offset = { 0, 0 };	// We don't want any offset to the render area because we want it to fit with the one of the attachments
				renderPassInfo.renderArea.extent = swapChainExtent;	// Define the area as the one defined in the swap chain

				std::array<VkClearValue, 1> clearValues = {};
				clearValues[0].color = { 1.0f, 0.8f, 1.0f, 1.0f };

				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(commandBuffersGUI[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

				// call to the gui to draw its values
				ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffersGUI[i]);

				vkCmdEndRenderPass(commandBuffersGUI[i]);
			}

			if (vkEndCommandBuffer(commandBuffersGUI[i]) != VK_SUCCESS) {	// Create the command buffer using the data from the struct 
																		// Throw a runtime error if something fails
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	// create a separate command pool that resets its values on drawing
	void createCommandPoolGUI(VkCommandPoolCreateFlags flags) {
		VkCommandPoolCreateInfo commandPoolCreateInfo = {};
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.queueFamilyIndex = findQueueFamilies(physicalDevice).graphicsFamily.value();
		commandPoolCreateInfo.flags = flags;

		if (commandPoolGUI)	// attempt to solve the memory leak
		{
			vkDeviceWaitIdle(device);
			vkDestroyCommandPool(device, commandPoolGUI, nullptr);
		}

		if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPoolGUI) != VK_SUCCESS) {
			throw std::runtime_error("Could not create graphics command pool");
		}
	}


	// First it will acquire an image from the swap chain,
	// Execute the command buffer using that image as attachment
	// Return the image to the swap chain for presentation 
	// All of this will be done asynchronously, but we will need semaphores and fences to control them
	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);	// Waits until either all or any of the fences to be signaled

		uint32_t imageIndex;
		// The first two parameters are the logical device and the swap chain from where we want to acquire images from
		// The third parameter is the time out that it has to get an image from the swap chain, 
		// The fourth parameter is the semaphore that should be signaled when the image is acquired
		// The last parameter is an output variable that will have the index of the retrieved image from the swap chain
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {	// If the swap chain has become incompatible with the window surface we recreate the swap chain
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {	// If there is any error with the swap chain acquiring an image throw a runtime error
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Update all the information of the uniforms
		updateUniformBuffer(imageIndex);
		updateLightUniformBuffer(imageIndex);	
		updateShadowUniformBuffer(imageIndex);
		
		if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {	// We check if a previous frame is using this image 
			vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);	// if it is we wait untill it finishes
		}
		imagesInFlight[imageIndex] = inFlightFences[currentFrame];	// Set the image to be in use in this frame

		createCommandPoolGUI(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		commandBuffersGUI.resize(swapChainImageViews.size());
		createCommandBuffersGUI();

		// create an array of command buffers to have the gui with the scene
		std::array<VkCommandBuffer, 2> submitCommandBuffers =
		{ commandBuffers[imageIndex], commandBuffersGUI[imageIndex] };

		// Handles queue submissions and synchronization
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;	// Set the type of the struct

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };	// Array containing one of the semaphores
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };	// Specify the stage of the pipeline after blending 
		submitInfo.waitSemaphoreCount = 1;	// It is the number of semaphores that will wait
		submitInfo.pWaitSemaphores = waitSemaphores;	// pointer to the array of semaphores that will wait
		submitInfo.pWaitDstStageMask = waitStages;	// Specify in which stage will they wait

		submitInfo.commandBufferCount = static_cast<uint32_t>(submitCommandBuffers.size()); // is the number of command buffers to execute in the batch
		submitInfo.pCommandBuffers = submitCommandBuffers.data();	// pointer to the command buffer to use

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };	// Specify an array containing one semaphore for the render finish
		submitInfo.signalSemaphoreCount = 1;	// is the number of semaphores that should be signaled after the execution of the command
		submitInfo.pSignalSemaphores = signalSemaphores;	// pointer to the array of semaphores that need to be signaled

		vkResetFences(device, 1, &inFlightFences[currentFrame]);	// Restore the fences to be unsignaled

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {	 // Submit the command buffer to the graphics queue
			throw std::runtime_error("failed to submit draw command buffer!");							 // Throw a runtime error if something fails
		}

		// Struct to configure the presentation of the frame back to the swap chain to be draw
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;	// Set the type of the struct

		presentInfo.waitSemaphoreCount = 1;	// Specify the number of semaphores that it should wait before starting
		presentInfo.pWaitSemaphores = signalSemaphores;	// Specify which semaphores it should wait before starting

		VkSwapchainKHR swapChains[] = { swapChain };	// Array containing the swap chain
		presentInfo.swapchainCount = 1;	// Specifies the number of swap chains to present images to 
		presentInfo.pSwapchains = swapChains; // Specifies the swap chains to present images to

		presentInfo.pImageIndices = &imageIndex;	// pointer to the image that will be present it to the swap chain

		result = vkQueuePresentKHR(presentQueue, &presentInfo);	// Submit the request to present an image to the swap chain using the data from the struct

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {	// If there is an error with the swap chain becomming incompatible with the window
																										// surface or the result is suboptimal or the framebuffer has been resized
																										// recreate the swap chain
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {	// If there is any error throw a runtime error 
			throw std::runtime_error("failed to present swap chain image!");
		}
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;	// update the current frame
	}

	// Wrapper of the bytecode to create a VkShaderModule object
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;	// set the type of the struct
		createInfo.codeSize = code.size();	//	size in bytes of the bytecode passed as argument
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());	// pointer to the code that will use the VkShaderModule object

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {	// Create the shader module with the data in the stuct create info. If there is an error
																								// throw a runtime error
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	//Choose which should be the color depth based on the format
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {	// search for the desired format and colorSpace
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				// if the format is BGRA and the colorspace supports SRGB return them
				return availableFormat;
			}
		}

		return availableFormats[0];	// return the first format in the array. As we don't have our desired we can pick the first one for example as there always will be one
	}

	// Indicates in which way will the images be shown to the screen
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) { // search for the desired present mode
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {	// the mode that we picked: represents the swap chain as a queue where the images are taken from the front 
																		// of the queue when the display is refreshed and if the queue is full the new images will replace
																		// the old ones allowing triple buffering
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;	// if the MailBox present mode is not available it will use the FIFO where when an image is ready 
											// it will be sent to be displayed causing tearing 
	}

	// Choose which resolution it will use for the images in the swap chain 
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {	// if a width of the window has been selected (different from MAX) it will use that one that fits the current window
			return capabilities.currentExtent;
		}
		else {	// if it is not the case we will have to guess which is the best resolution
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);	// Get the size of the window with the help of GLFW

			VkExtent2D actualExtent = {	// fit the information in a struct
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));	// get the width between the limits and the actual extent
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height)); // get the height between the limits and the actual extent

			return actualExtent;	// return the final selected resolution
		}
	}

	// Query the information need it for filling in the details for the swap chain
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);	// retrieves the basic surface capabilities

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);	// get the number of suface formats

		if (formatCount != 0) {	// if there are any format
			details.formats.resize(formatCount);	// resize an actual array of the struct to have the size of the number of surface formats
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());	// set the data of each format inside of the array
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);	// get the number of present modes

		if (presentModeCount != 0) {	// if there are any present mode
			details.presentModes.resize(presentModeCount);	// resize an actual array of the struct to have the size of the number of present modes
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());	// set the data of each present mode inside of the array
		}

		return details;	// return the struct with all the data related to the details of the swap chain
	}

	// Checks if the graphics card "device" is suitable for the purpose of the program that will be built
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device); // ask if the device is able to proccess the commands that we want

		bool extensionsSupported = checkDeviceExtensionSupport(device);	// checks if the device support all the required extensions

		bool swapChainAdequate = false;
		if (extensionsSupported) {	// if it support all required extensions
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);	// retrieve the details of the swap chain
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();	// Set the swap chain to be adequate 
																												// if there is at least a format and a present mode
		}
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;	// if everything is supported and adequate and we have the indices of the queues return true
	}

	// check if the device is able to provide the different extensions required 
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);	// get the number of extensions in the system

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);	// create an array to store the data of every extension
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());	// fill in the data of every extension in the system into the array 

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());	// create a set of strings with the extensions that the device supports

		for (const auto& extension : availableExtensions) {	// for each extension of the system that the device supports
			requiredExtensions.erase(extension.extensionName);	// erase the extensions from the list of required extensions
		}

		return requiredExtensions.empty();	// if the list of requiered extensions is empty it means that all of them are supported
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;	// It will hold the queues indexes 

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);	// look for the number of queues available to the system

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);	// array that holds the details of every queue
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data()); // fill in the array with the values of every queue

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {	// check for every queue available to the system
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {	// if there is the one related to the graphics commands
				indices.graphicsFamily = i;	// assing the index of the loop to the graphics queue
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);	// search for a queue able to present to our window surface

			if (presentSupport) {	// if the queue for presenting was found
				indices.presentFamily = i;	// set the index to the one in the loop
			}

			if (indices.isComplete()) {	// Check if we have already find all the queues that we wanted so we don't need to keep looping.
				break;
			}

			i++;
		}

		return indices;	// return the updated values of the queues
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	// GLFW get for us the extensions that are need it

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);	// vector that will hold the extensions that our application is going to have

		if (enableValidationLayers) {	// if validation layers are activated
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	// add this extension to the extension list
		}

		return extensions;	// return all the extensions 
	}

	// Checks if all requested validations layer are available to the program
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	// We get the number of layer properties available to the system

		std::vector<VkLayerProperties> availableLayers(layerCount);	// create a vector with the size of the number of layer properties available, in order
																	// to hold the data of each of them
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());	// fill in the data of each available layer properties in the vector.

		// Checks if every desired validation layer to the system is in the list of the available layers to the system
		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {	// if any of the layers fail it returns false meaning that not all desired validation layers are available.
				return false;
			}
		}

		return true;	// otherwise everything is as we would like and it return true (success)
	}

	// It will read the binary files for the shaders from end to beginning
	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);	// open the file from the end (ate) and in binary format to avoid text transformations

		if (!file.is_open()) {	// if the file is not correctly opened throw a runtime error
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();	// As we started the file from the end the pointer of the file tell us its size
		std::vector<char> buffer(fileSize);	// Create the buffer with the size of the input file 

		file.seekg(0);	// Go to the beginning of the file
		file.read(buffer.data(), fileSize);	// get the data from the file that was opened and throw it into the buffer

		file.close();	// close the file

		return buffer;
	}

	// the VKAPI_ATTR, VKAPI_CALL are need it for Vulkan to correctly call it
	// the first parameter tells the severity of the message. ie. error, warning, diagnostic...
	// the type can be: unrelated to performance nor specificatione erro, violates specification, non optimal performance
	// pCallbackData contains details of the message itself. If it is a string, array of Vulkan objects or number of objects in an array
	// pUserData a pointer to allow passing data to it
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;	// output the message data

		return VK_FALSE;	// this return tells if the program should be aborted
	}
};

int main() {
	HelloTriangleApplication app;

	try {

		app.run();	// execute the initialization, the loop, and the cleanup
	}
	catch (const std::exception& e) {	// Catch any type of error
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}