#ifndef __DOC_SCANNER_H__
#define __DOC_SCANNER_H__

#include <Python.h>
#include <structmember.h>
#include "DynamsoftDocumentNormalizer.h"
#include "document_result.h"
#include "normalized_image.h"
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <functional>

class Task
{
public:
    std::function<void()> func;
    unsigned char* buffer;
};

class WorkerThread
{
public:
    std::mutex m;
    std::condition_variable cv;
    std::queue<Task> tasks = {};
    volatile bool running;
	std::thread t;
};

typedef struct
{
    PyObject_HEAD 
    void *handler;
    PyObject *callback;
    WorkerThread *worker;
} DynamsoftDocumentScanner;

void clearTasks(DynamsoftDocumentScanner *self)
{
    if (self->worker->tasks.size() > 0)
    {
        for (int i = 0; i < self->worker->tasks.size(); i++)
        {
            free(self->worker->tasks.front().buffer);
            self->worker->tasks.pop();
        }
    }
}

void clear(DynamsoftDocumentScanner *self)
{
    if (self->callback)
    {
        Py_XDECREF(self->callback);
        self->callback = NULL;
    }

    if (self->worker)
    {
        self->worker->running = false;
        clearTasks(self);
        self->worker->cv.notify_one();
        self->worker->t.join();
        delete self->worker;
        self->worker = NULL;
        printf("Quit native thread.\n");
    }
}

static int DynamsoftDocumentScanner_clear(DynamsoftDocumentScanner *self)
{
    clear(self);

    if (self->handler)
    {
        DDN_DestroyInstance(self->handler);
        self->handler = NULL;
    }
    return 0;
}

static void DynamsoftDocumentScanner_dealloc(DynamsoftDocumentScanner *self)
{
    DynamsoftDocumentScanner_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DynamsoftDocumentScanner_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DynamsoftDocumentScanner *self;

    self = (DynamsoftDocumentScanner *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->handler = DDN_CreateInstance();
        self->worker = NULL;
        self->callback = NULL;
    }

    return (PyObject *)self;
}

PyObject *createPyList(DetectedQuadResultArray *pResults)
{
    // Create a Python object to store results
    PyObject *list = PyList_New(0);

    if (pResults)
    {
        int count = pResults->resultsCount;
    
        for (int i = 0; i < count; i++)
        {
            DetectedQuadResult *quadResult = pResults->detectedQuadResults[i];
            int confidence = quadResult->confidenceAsDocumentBoundary;
            DM_Point *points = quadResult->location->points;
            int x1 = points[0].coordinate[0];
            int y1 = points[0].coordinate[1];
            int x2 = points[1].coordinate[0];
            int y2 = points[1].coordinate[1];
            int x3 = points[2].coordinate[0];
            int y3 = points[2].coordinate[1];
            int x4 = points[3].coordinate[0];
            int y4 = points[3].coordinate[1];
            
            DocumentResult *result = PyObject_New(DocumentResult, &DocumentResultType);
            result->confidence = Py_BuildValue("i", confidence);
            result->x1 = Py_BuildValue("i", x1);
            result->y1 = Py_BuildValue("i", y1);
            result->x2 = Py_BuildValue("i", x2);
            result->y2 = Py_BuildValue("i", y2);
            result->x3 = Py_BuildValue("i", x3);
            result->y3 = Py_BuildValue("i", y3);
            result->x4 = Py_BuildValue("i", x4);
            result->y4 = Py_BuildValue("i", y4);

            PyList_Append(list, (PyObject *)result);
        }
    }

    return list;
}

/**
 * Recognize document from image files.
 *
 * @param string filename
 *
 * @return DocumentResult list
 */
static PyObject *detectFile(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    char *pFileName; // File name
    if (!PyArg_ParseTuple(args, "s", &pFileName))
    {
        return NULL;
    }

    DetectedQuadResultArray *pResults = NULL;
    
    int ret = DDN_DetectQuadFromFile(self->handler, pFileName, "", &pResults);
    if (ret)
    {
        printf("Detection error: %s\n", DC_GetErrorString(ret));
    }

    PyObject *list = createPyList(pResults);

    // Release memory
    if (pResults != NULL)
        DDN_FreeDetectedQuadResultArray(&pResults);

    return list;
}

/**
 * Recognize document from OpenCV Mat.
 *
 * @param Mat image
 *
 * @return DocumentResult list
 */
static PyObject *detectMat(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
        return NULL;

    Py_buffer *view;
    int nd;
    PyObject *memoryview = PyMemoryView_FromObject(o);
    if (memoryview == NULL)
    {
        PyErr_Clear();
        return NULL;
    }

    view = PyMemoryView_GET_BUFFER(memoryview);
    char *buffer = (char *)view->buf;
    nd = view->ndim;
    int len = view->len;
    int stride = view->strides[0];
    int width = view->strides[0] / view->strides[1];
    int height = len / stride;

    ImagePixelFormat format = IPF_RGB_888;

    if (width == stride)
    {
        format = IPF_GRAYSCALED;
    }
    else if (width * 3 == stride)
    {
        format = IPF_RGB_888;
    }
    else if (width * 4 == stride)
    {
        format = IPF_ARGB_8888;
    }

    ImageData data;
    data.bytes = (unsigned char *)buffer;
    data.width = width;
    data.height = height;
    data.stride = stride;
    data.format = format;
    data.bytesLength = len;

    DetectedQuadResultArray *pResults = NULL;
    
    int ret = DDN_DetectQuadFromBuffer(self->handler, &data, "", &pResults);
    if (ret)
    {
        printf("Detection error: %s\n", DC_GetErrorString(ret));
    }

    PyObject *list = createPyList(pResults);

    // Release memory
    if (pResults != NULL)
        DDN_FreeDetectedQuadResultArray(&pResults);

    Py_DECREF(memoryview);

    return list;
}

void onResultReady(DynamsoftDocumentScanner *self, DetectedQuadResultArray *pResults)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject *list = createPyList(pResults);
    PyObject *result = PyObject_CallFunction(self->callback, "O", list);
    if (result != NULL)
        Py_DECREF(result);

    PyGILState_Release(gstate);
}

void scan(DynamsoftDocumentScanner *self, unsigned char *buffer, int width, int height, int stride, ImagePixelFormat format, int len)
{
    ImageData data;
    data.bytes = buffer;
    data.width = width;
    data.height = height;
    data.stride = stride;
    data.format = format;
    data.bytesLength = len;

    DetectedQuadResultArray *pResults = NULL;
    int ret = DDN_DetectQuadFromBuffer(self->handler, &data, "", &pResults);
    if (ret)
    {
        printf("Detection error: %s\n", DC_GetErrorString(ret));
    }

    free(buffer);
    if (self->callback != NULL)
    {
        onResultReady(self, pResults);
    }

    // Release memory
    if (pResults != NULL)
        DDN_FreeDetectedQuadResultArray(&pResults);
}

/**
 * Recognize document from OpenCV Mat asynchronously.
 *
 * @param Mat image
 *
 */
static PyObject *detectMatAsync(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
        return Py_BuildValue("i", -1);

    Py_buffer *view;
    int nd;
    PyObject *memoryview = PyMemoryView_FromObject(o);
    if (memoryview == NULL)
    {
        PyErr_Clear();
        return Py_BuildValue("i", -1);
    }

    view = PyMemoryView_GET_BUFFER(memoryview);
    char *buffer = (char *)view->buf;
    nd = view->ndim;
    int len = view->len;
    int stride = view->strides[0];
    int width = view->strides[0] / view->strides[1];
    int height = len / stride;

    ImagePixelFormat format = IPF_RGB_888;

    if (width == stride)
    {
        format = IPF_GRAYSCALED;
    }
    else if (width * 3 == stride)
    {
        format = IPF_RGB_888;
    }
    else if (width * 4 == stride)
    {
        format = IPF_ARGB_8888;
    }

    unsigned char *data = (unsigned char *)malloc(len);
    memcpy(data, buffer, len);

    std::unique_lock<std::mutex> lk(self->worker->m);
    clearTasks(self);
    std::function<void()> task_function = std::bind(scan, self, data, width, height, stride, format, len);
    Task task;
    task.func = task_function;
    task.buffer = data;
    self->worker->tasks.push(task);
    self->worker->cv.notify_one();
    lk.unlock();
    
    Py_DECREF(memoryview);
    return Py_BuildValue("i", 0);
}

void run(DynamsoftDocumentScanner *self)
{
    while (self->worker->running)
    {
        std::function<void()> task;
        std::unique_lock<std::mutex> lk(self->worker->m);
        self->worker->cv.wait(lk, [&]
                              { return !self->worker->tasks.empty() || !self->worker->running; });
        if (!self->worker->running)
		{
			break;
		}
        task = std::move(self->worker->tasks.front().func);
        self->worker->tasks.pop();
        lk.unlock();

        task();
    }
}

/**
 * Register callback function to receive document decoding result asynchronously.
 */
static PyObject *addAsyncListener(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    PyObject *callback = NULL;
    if (!PyArg_ParseTuple(args, "O", &callback))
    {
        return NULL;
    }

    if (!PyCallable_Check(callback))
    {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    else
    {
        Py_XINCREF(callback);       /* Add a reference to new callback */
        Py_XDECREF(self->callback); /* Dispose of previous callback */
        self->callback = callback;
    }

    if (self->worker == NULL)
    {
        self->worker = new WorkerThread();
        self->worker->running = true;
        self->worker->t = std::thread(&run, self);
    }

    return Py_BuildValue("i", 0);
}

/**
 * Clear native thread and tasks.
 */
static PyObject *clearAsyncListener(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;
    clear(self);
    return Py_BuildValue("i", 0);
}

static PyObject *setParameters(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    const char*params;
    if (!PyArg_ParseTuple(args, "s", &params))
    {
        return NULL;
    }

    char errorMsgBuffer[512];
    int ret = DDN_InitRuntimeSettingsFromString(self->handler, params, errorMsgBuffer, 512);
	printf("Init runtime settings: %s\n", errorMsgBuffer);

    return Py_BuildValue("i", ret);
}

PyObject *createNormalizedImage(NormalizedImageResult* normalizedResult)
{
    if (normalizedResult == NULL)
    {
        return NULL;
    }

    ImageData *imageData = normalizedResult->image;
    NormalizedImage *ni = PyObject_New(NormalizedImage, &NormalizedImageType);
    ni->bytearray = PyByteArray_FromStringAndSize((const char *)imageData->bytes, imageData->bytesLength);
    ni->length = Py_BuildValue("i", imageData->bytesLength);
    ni->width = Py_BuildValue("i", imageData->width);
    ni->height = Py_BuildValue("i", imageData->height);
    ni->stride = Py_BuildValue("i", imageData->stride);
    ni->format = Py_BuildValue("i", imageData->format);
    ni->normalizedResult = normalizedResult;
    return (PyObject *)ni;
}
/**
 * Normalize the document.
 *
 * @param string filePath
 *
 * @return Normalized document image
 */
static PyObject *normalizeFile(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    char *pFileName; 
    int x1, y1, x2, y2, x3, y3, x4, y4;
    if (!PyArg_ParseTuple(args, "siiiiiiii", &pFileName, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4))
        return NULL;

    Quadrilateral quad;
    quad.points[0].coordinate[0] = x1;
    quad.points[0].coordinate[1] = y1;
    quad.points[1].coordinate[0] = x2;
    quad.points[1].coordinate[1] = y2;
    quad.points[2].coordinate[0] = x3;
    quad.points[2].coordinate[1] = y3;
    quad.points[3].coordinate[0] = x4;
    quad.points[3].coordinate[1] = y4;

    NormalizedImageResult* normalizedResult = NULL;
    
    int errorCode = DDN_NormalizeFile(self->handler, pFileName, "", &quad, &normalizedResult);
    if (errorCode != DM_OK)
        printf("%s\r\n", DC_GetErrorString(errorCode));

    PyObject *normalizedImage = createNormalizedImage(normalizedResult);

    return normalizedImage;
}

/**
 * Normalize the document.
 *
 * @param Mat image
 *
 * @return Normalized document image
 */
static PyObject *normalizeBuffer(PyObject *obj, PyObject *args)
{
    DynamsoftDocumentScanner *self = (DynamsoftDocumentScanner *)obj;

    PyObject *o;
    int x1, y1, x2, y2, x3, y3, x4, y4;
    if (!PyArg_ParseTuple(args, "Oiiiiiiii", &o, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4))
        return NULL;

    Py_buffer *view;
    int nd;
    PyObject *memoryview = PyMemoryView_FromObject(o);
    if (memoryview == NULL)
    {
        PyErr_Clear();
        return NULL;
    }

    view = PyMemoryView_GET_BUFFER(memoryview);
    char *buffer = (char *)view->buf;
    nd = view->ndim;
    int len = view->len;
    int stride = view->strides[0];
    int width = view->strides[0] / view->strides[1];
    int height = len / stride;

    ImagePixelFormat format = IPF_RGB_888;

    if (width == stride)
    {
        format = IPF_GRAYSCALED;
    }
    else if (width * 3 == stride)
    {
        format = IPF_RGB_888;
    }
    else if (width * 4 == stride)
    {
        format = IPF_ARGB_8888;
    }

    ImageData data;
    data.bytes = (unsigned char *)buffer;
    data.width = width;
    data.height = height;
    data.stride = stride;
    data.format = format;
    data.bytesLength = len;

    Quadrilateral quad;
    quad.points[0].coordinate[0] = x1;
    quad.points[0].coordinate[1] = y1;
    quad.points[1].coordinate[0] = x2;
    quad.points[1].coordinate[1] = y2;
    quad.points[2].coordinate[0] = x3;
    quad.points[2].coordinate[1] = y3;
    quad.points[3].coordinate[0] = x4;
    quad.points[3].coordinate[1] = y4;

    NormalizedImageResult* normalizedResult = NULL;
    int errorCode = DDN_NormalizeBuffer(self->handler, &data, "", &quad, &normalizedResult);
    if (errorCode != DM_OK)
        printf("%s\r\n", DC_GetErrorString(errorCode));

    PyObject *normalizedImage = createNormalizedImage(normalizedResult);

    Py_DECREF(memoryview);

   return normalizedImage;
}

static PyMethodDef instance_methods[] = {
    {"detectFile", detectFile, METH_VARARGS, NULL},
    {"detectMat", detectMat, METH_VARARGS, NULL},
    {"addAsyncListener", addAsyncListener, METH_VARARGS, NULL},
    {"detectMatAsync", detectMatAsync, METH_VARARGS, NULL},
    {"setParameters", setParameters, METH_VARARGS, NULL},
    {"normalizeFile", normalizeFile, METH_VARARGS, NULL},
    {"normalizeBuffer", normalizeBuffer, METH_VARARGS, NULL},
    {"clearAsyncListener", clearAsyncListener, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
    };

static PyTypeObject DynamsoftDocumentScannerType = {
    PyVarObject_HEAD_INIT(NULL, 0) "docscanner.DynamsoftDocumentScanner", /* tp_name */
    sizeof(DynamsoftDocumentScanner),                                     /* tp_basicsize */
    0,                                                              /* tp_itemsize */
    (destructor)DynamsoftDocumentScanner_dealloc,                         /* tp_dealloc */
    0,                                                              /* tp_print */
    0,                                                              /* tp_getattr */
    0,                                                              /* tp_setattr */
    0,                                                              /* tp_reserved */
    0,                                                              /* tp_repr */
    0,                                                              /* tp_as_number */
    0,                                                              /* tp_as_sequence */
    0,                                                              /* tp_as_mapping */
    0,                                                              /* tp_hash  */
    0,                                                              /* tp_call */
    0,                                                              /* tp_str */
    PyObject_GenericGetAttr,                                        /* tp_getattro */
    PyObject_GenericSetAttr,                                        /* tp_setattro */
    0,                                                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                       /*tp_flags*/
    "DynamsoftDocumentScanner",                                           /* tp_doc */
    0,                                                              /* tp_traverse */
    0,                                                              /* tp_clear */
    0,                                                              /* tp_richcompare */
    0,                                                              /* tp_weaklistoffset */
    0,                                                              /* tp_iter */
    0,                                                              /* tp_iternext */
    instance_methods,                                               /* tp_methods */
    0,                                                              /* tp_members */
    0,                                                              /* tp_getset */
    0,                                                              /* tp_base */
    0,                                                              /* tp_dict */
    0,                                                              /* tp_descr_get */
    0,                                                              /* tp_descr_set */
    0,                                                              /* tp_dictoffset */
    0,                                                              /* tp_init */
    0,                                                              /* tp_alloc */
    DynamsoftDocumentScanner_new,                                         /* tp_new */
};

#endif