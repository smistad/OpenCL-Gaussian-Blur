#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"

using namespace cl;

float * createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}


int main(int argc, char ** argv) {
    // Load image
    SIPL::Image<float> * image = new SIPL::Image<float>("images/sunset.jpg");

    // Create OpenCL context
    Context context = createCLContextFromArguments(argc, argv);

    // Compile OpenCL code
    Program program = buildProgramFromSource(context, "gaussian_blur.cl");
    
    // Select device and create a command queue for it
    VECTOR_CLASS<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CommandQueue queue = CommandQueue(context, devices[0]);

    // Create an OpenCL Image / texture and transfer data to the device
    Image2D clImage = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), image->getWidth(), image->getHeight(), 0, (void*)image->getData());

    // Create a buffer for the result
    Buffer clResult = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*image->getWidth()*image->getHeight());

    // Create Gaussian mask
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);
    
    // Create buffer for mask and transfer it to the device
    Buffer clMask = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1), mask);

    // Run Gaussian kernel
    Kernel gaussianBlur = Kernel(program, "gaussian_blur");
    gaussianBlur.setArg(0, clImage);
    gaussianBlur.setArg(1, clMask);
    gaussianBlur.setArg(2, clResult);
    gaussianBlur.setArg(3, maskSize);

    queue.enqueueNDRangeKernel(
        gaussianBlur,
        NullRange,
        NDRange(image->getWidth(), image->getHeight()),
        NullRange
    );

    // Transfer image back to host
    float* data = new float[image->getWidth()*image->getHeight()];
    queue.enqueueReadBuffer(clResult, CL_TRUE, 0, sizeof(float)*image->getWidth()*image->getHeight(), data); 
    image->setData(data);

    // Save image to disk
    image->save("images/result.jpg", "jpeg");
    image->display();
}
