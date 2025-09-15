import xml.etree.ElementTree as ET

try:
    ET.parse("Sample.xml")
    print("XML is fine ✅")
except ET.ParseError as e:
    print("XML problem:", e)