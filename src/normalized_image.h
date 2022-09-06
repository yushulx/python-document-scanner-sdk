#ifndef __NORMALIZED_IMAGE_H__
#define __NORMALIZED_IMAGE_H__

#include <Python.h>
#include <structmember.h>

// https://docs.python.org/3/c-api/typeobj.html#typedef-examples
typedef struct 
{
	PyObject_HEAD
    PyObject *bytearray;
	PyObject *length;
	PyObject *width;
	PyObject *height;
	PyObject *stride;
	PyObject *format;
    NormalizedImageResult* normalizedResult;
} NormalizedImage;

static void NormalizedImage_dealloc(NormalizedImage *self)
{
	if (self->bytearray) Py_DECREF(self->bytearray);
    if (self->length) Py_DECREF(self->length);
    if (self->width) Py_DECREF(self->width);
    if (self->height) Py_DECREF(self->height);
    if (self->stride) Py_DECREF(self->stride);
    if (self->format) Py_DECREF(self->format);
    if (self->normalizedResult) DDN_FreeNormalizedImageResult(&self->normalizedResult);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *NormalizedImage_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NormalizedImage *self;

    self = (NormalizedImage *)type->tp_alloc(type, 0);
    self->normalizedResult = NULL;
    return (PyObject *)self;
}

static PyObject *save(PyObject *obj, PyObject *args)
{
    NormalizedImage *self = (NormalizedImage *)obj;

    char *pFileName; // File name
    if (!PyArg_ParseTuple(args, "s", &pFileName))
    {
        return NULL;
    }

    if (self->normalizedResult)
    {
        DDN_SaveImageDataToFile(self->normalizedResult->image, pFileName);
        printf("Save image to file: %s\n", pFileName);
        return Py_BuildValue("i", 0);
    }

    return Py_BuildValue("i", -1);
}

static PyObject *recycle(PyObject *obj, PyObject *args)
{
    NormalizedImage *self = (NormalizedImage *)obj;

    if (self->normalizedResult)
    {
        DDN_FreeNormalizedImageResult(&self->normalizedResult);
        self->normalizedResult = NULL;
    }

    return Py_BuildValue("i", 0);
}

static PyMemberDef NormalizedImage_members[] = {
    {"bytearray", T_OBJECT_EX, offsetof(NormalizedImage, bytearray), 0, "bytearray"},
    {"length", T_OBJECT_EX, offsetof(NormalizedImage, length), 0, "length"},
    {"width", T_OBJECT_EX, offsetof(NormalizedImage, width), 0, "width"},
    {"height", T_OBJECT_EX, offsetof(NormalizedImage, height), 0, "height"},
    {"stride", T_OBJECT_EX, offsetof(NormalizedImage, stride), 0, "stride"},
    {"format", T_OBJECT_EX, offsetof(NormalizedImage, format), 0, "format"},
    {NULL}  /* Sentinel */
};

static PyMethodDef ni_instance_methods[] = {
    {"save", save, METH_VARARGS, NULL},
    {"recycle", recycle, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject NormalizedImageType = {
    PyVarObject_HEAD_INIT(NULL, 0) "docscanner.NormalizedImage", /* tp_name */
    sizeof(NormalizedImage),                                       /* tp_basicsize */
    0,                                                           /* tp_itemsize */
    (destructor)NormalizedImage_dealloc,                           /* tp_dealloc */
    0,                                                           /* tp_print */
    0,                                                           /* tp_getattr */
    0,                                                           /* tp_setattr */
    0,                                                           /* tp_reserved */
    0,                                                           /* tp_repr */
    0,                                                           /* tp_as_number */
    0,                                                           /* tp_as_sequence */
    0,                                                           /* tp_as_mapping */
    0,                                                           /* tp_hash  */
    0,                                                           /* tp_call */
    0,                                                           /* tp_str */
    PyObject_GenericGetAttr,                                     /* tp_getattro */
    PyObject_GenericSetAttr,                                     /* tp_setattro */
    0,                                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                    /*tp_flags*/
    "NormalizedImage",                                             /* tp_doc */
    0,                                                           /* tp_traverse */
    0,                                                           /* tp_clear */
    0,                                                           /* tp_richcompare */
    0,                                                           /* tp_weaklistoffset */
    0,                                                           /* tp_iter */
    0,                                                           /* tp_iternext */
    ni_instance_methods,                                                           /* tp_methods */
    NormalizedImage_members,                                       /* tp_members */
    0,                                                           /* tp_getset */
    0,                                                           /* tp_base */
    0,                                                           /* tp_dict */
    0,                                                           /* tp_descr_get */
    0,                                                           /* tp_descr_set */
    0,                                                           /* tp_dictoffset */
    0,                                                           /* tp_init */
    0,                                                           /* tp_alloc */
    NormalizedImage_new,                                           /* tp_new */
};

#endif