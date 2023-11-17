from rest_framework import serializers


class LayoutNerInputSerializer(serializers.Serializer):
    image = serializers.ImageField()
    # Add other fields as needed (tokens, bboxes, etc.)

class LayoutNerOutputSerializer(serializers.Serializer):
    result_table = serializers.CharField()
    # Add other fields as needed
