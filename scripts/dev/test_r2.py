#!/usr/bin/env python3
"""SigV4 signer of scripts/dev/r2.py against AWS's published S3 example
(GET Object, "Signature Calculations for the Authorization Header",
docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html).

  python3 scripts/dev/test_r2.py
"""

import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2  # noqa: E402

EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class SigV4(unittest.TestCase):
    def test_aws_get_object_example(self):
        now = datetime.datetime(2013, 5, 24, 0, 0, 0, tzinfo=datetime.timezone.utc)
        h = r2.sign("GET", "examplebucket.s3.amazonaws.com", "/test.txt", {}, {"Range": "bytes=0-9"}, EMPTY,
                    "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", now, region="us-east-1")
        self.assertEqual(
            h["authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
            "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
            "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
        )

    def test_aws_list_objects_example(self):
        # GET Bucket (List Objects) example from the same page: query string signed.
        now = datetime.datetime(2013, 5, 24, 0, 0, 0, tzinfo=datetime.timezone.utc)
        h = r2.sign("GET", "examplebucket.s3.amazonaws.com", "/", {"max-keys": "2", "prefix": "J"}, {}, EMPTY,
                    "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", now, region="us-east-1")
        self.assertTrue(h["authorization"].endswith(
            "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"))

    def test_key_quoting(self):
        self.assertEqual(r2.quote_path("/b/artifacts/abc/serve cuda.tar.gz"), "/b/artifacts/abc/serve%20cuda.tar.gz")


if __name__ == "__main__":
    unittest.main()
