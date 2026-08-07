class RawSms {
  final String body;
  final String sender;
  final DateTime date;
  final int timestampMs;

  const RawSms({
    required this.body,
    required this.sender,
    required this.date,
    required this.timestampMs,
  });
}

class SmsReadResult {
  final bool isSuccess;
  final List<RawSms> messages;

  const SmsReadResult({
    required this.isSuccess,
    required this.messages,
  });
}

class SmsService {
  SmsService._();
  static final SmsService instance = SmsService._();

  Future<bool> hasPermission() async => true;

  Future<bool> requestPermission() async => true;

  Future<SmsReadResult> readAllSms() async {
    return const SmsReadResult(isSuccess: true, messages: []);
  }

  Future<SmsReadResult> scanLatestSms({required DateTime since}) async {
    return const SmsReadResult(isSuccess: true, messages: []);
  }
}
