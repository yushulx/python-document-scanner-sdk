// Python includes
#include <Python.h>

// STD includes
#include <stdio.h>

#include "document_scanner.h"

#define INITERROR return NULL

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static PyObject *
error_out(PyObject *m)
{
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

#define DBR_NO_MEMORY 0
#define DBR_SUCCESS 1

// #define LOG_OFF

#ifdef LOG_OFF

#define printf(MESSAGE, __VA_ARGS__)

#endif

#define DEFAULT_MEMORY_SIZE 4096

static PyObject *createInstance(PyObject *obj, PyObject *args)
{
    if (PyType_Ready(&DynamsoftDocumentScannerType) < 0)
         INITERROR;

    DynamsoftDocumentScanner* reader = PyObject_New(DynamsoftDocumentScanner, &DynamsoftDocumentScannerType);
    reader->handler = DDN_CreateInstance();
    reader->worker = NULL;
    reader->callback = NULL;
    return (PyObject *)reader;
}
    
static PyObject *initLicense(PyObject *obj, PyObject *args)
{
    char *pszLicense;
    if (!PyArg_ParseTuple(args, "s", &pszLicense))
    {
        return NULL;
    }

    char errorMsgBuffer[512];
	// Click https://www.dynamsoft.com/customer/license/trialLicense/?product=dbr to get a trial license.
	int ret = DC_InitLicense(pszLicense, errorMsgBuffer, 512);
	printf("DC_InitLicense: %s\n", errorMsgBuffer);

    return Py_BuildValue("i", ret);
}

static PyMethodDef docscanner_methods[] = {
  {"initLicense", initLicense, METH_VARARGS, "Set license to activate the SDK"},
  {"createInstance", createInstance, METH_VARARGS, "Create Dynamsoft MRZ Reader object"},
  {NULL, NULL, 0, NULL}       
};

static struct PyModuleDef docscanner_module_def = {
  PyModuleDef_HEAD_INIT,
  "docscanner",
  "Internal \"docscanner\" module",
  -1,
  docscanner_methods
};

// https://docs.python.org/3/c-api/module.html
// https://docs.python.org/3/c-api/dict.html
PyMODINIT_FUNC PyInit_docscanner(void)
{
	PyObject *module = PyModule_Create(&docscanner_module_def);
    if (module == NULL)
        INITERROR;

    
    if (PyType_Ready(&DynamsoftDocumentScannerType) < 0)
       INITERROR;

    Py_INCREF(&DynamsoftDocumentScannerType);
    PyModule_AddObject(module, "DynamsoftDocumentScanner", (PyObject *)&DynamsoftDocumentScannerType);
    
    if (PyType_Ready(&DocumentResultType) < 0)
       INITERROR;

    Py_INCREF(&DocumentResultType);
    PyModule_AddObject(module, "DocumentResult", (PyObject *)&DocumentResultType);

	PyModule_AddStringConstant(module, "version", DDN_GetVersion());
    return module;
}

