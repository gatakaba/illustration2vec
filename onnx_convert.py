import os
import onnx
import coremltools
import onnxmltools

# Convert Caffe model into ONNX
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb


def caffe_to_onnx(proto_file, input_caffe_path):
    output_coreml_model = 'model.mlmodel'  # intermediate file
    if os.path.exists(output_coreml_model):
        raise FileExistsError('model.mlmodel already exists')

    # Convert Caffe model to CoreML 
    coreml_model = coremltools.converters.caffe.convert(
        (input_caffe_path, proto_file))
    # Save CoreML model
    coreml_model.save(output_coreml_model)
    # Load a Core ML model
    coreml_model = coremltools.utils.load_spec(output_coreml_model)
    # Convert the Core ML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model)
    # Remove Core ML model
    os.remove(output_coreml_model)
    return onnx_model


# Save tag prediction model
onnx_tag_model = caffe_to_onnx(
    'illust2vec_tag.prototxt', 'illust2vec_tag_ver200.caffemodel')
onnxmltools.utils.save_model(onnx_tag_model, 'illust2vec_tag_ver200.onnx')

# Save feature vector extraction model
onnx_model = caffe_to_onnx('illust2vec.prototxt',
                           'illust2vec_ver200.caffemodel')
# make 'encode1' layer accessible from ONNX
# https://github.com/microsoft/onnxruntime/issues/2119
intermediate_tensor_name = 'encode1'
intermediate_layer_value_info = onnx.helper.ValueInfoProto()
intermediate_layer_value_info.name = intermediate_tensor_name
onnx_model.graph.output.extend([intermediate_layer_value_info])
onnx.save(onnx_model, 'illust2vec_ver200.onnx')
