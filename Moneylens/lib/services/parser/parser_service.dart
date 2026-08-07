import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:uuid/uuid.dart';
import 'package:moneylens/models/transaction_model.dart';
import 'package:moneylens/services/sms/sms_service.dart';

/// Result of attempting to parse one SMS
class ParseResult {
  final TransactionModel? transaction;
  final bool success;
  final String? failReason;

  const ParseResult._({
    this.transaction,
    required this.success,
    this.failReason,
  });

  factory ParseResult.ok(TransactionModel tx) =>
      ParseResult._(transaction: tx, success: true);

  factory ParseResult.failed(String reason) =>
      ParseResult._(success: false, failReason: reason);
}

// ─────────────────────────────────────────────────────────────────────────────

class ParserService {
  ParserService._();
  static final ParserService instance = ParserService._();

  final _uuid = const Uuid();

  // ─── Public API ───────────────────────────────────────────────────────────

  /// Parse a single RawSms → ParseResult.
  ParseResult parse(RawSms sms) {
    try {
      final body = sms.body.trim();

      final amount = extractAmount(body);
      if (amount == null || amount <= 0) {
        return ParseResult.failed('No valid amount found');
      }

      final type = extractTransactionType(body);
      final merchant = extractMerchant(body);
      final bank = extractBank(body, sms.sender);
      final date = extractDate(body) ?? sms.date;
      final hash = _generateHash(body, sms.timestampMs);

      return ParseResult.ok(
        TransactionModel(
          id: _uuid.v4(),
          amount: amount,
          merchant: merchant,
          category: 'Others', // filled in by CategorizerService after parsing
          bank: bank,
          transactionType: type,
          date: date,
          smsBody: body,
          smsHash: hash,
          createdAt: DateTime.now(),
        ),
      );
    } catch (e) {
      return ParseResult.failed('Exception: $e');
    }
  }

  /// Parse a list of RawSms — skips failures silently (as per TDD).
  List<TransactionModel> parseAll(List<RawSms> smsList) {
    final results = <TransactionModel>[];
    for (final sms in smsList) {
      final result = parse(sms);
      if (result.success && result.transaction != null) {
        results.add(result.transaction!);
      }
    }
    return results;
  }

  // ─── Amount Extraction ────────────────────────────────────────────────────

  /// Extracts the transaction amount from SMS body.
  /// Handles: Rs.420, Rs 420, INR 420, ₹420, ₹ 4,20,000.50
  double? extractAmount(String body) {
    const patterns = [
      // ₹4,20,000.50 or ₹420
      r'₹\s*([0-9,]+(?:\.[0-9]{1,2})?)',
      // Rs.420 or Rs 420 or Rs. 4,200
      r'[Rr][Ss]\.?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
      // INR 420
      r'INR\s*([0-9,]+(?:\.[0-9]{1,2})?)',
    ];

    for (final pattern in patterns) {
      final match = RegExp(pattern).firstMatch(body);
      if (match != null) {
        final raw = match.group(1)!.replaceAll(',', '');
        final value = double.tryParse(raw);
        if (value != null && value > 0) return value;
      }
    }
    return null;
  }

  // ─── Transaction Type ─────────────────────────────────────────────────────

  /// Determines debit or credit from SMS keywords.
  TransactionType extractTransactionType(String body) {
    final lower = body.toLowerCase();

    // Credit signals — check first because some SMS have both words
    const creditKeywords = [
      'credited',
      'received',
      'deposited',
      'refund',
      'cashback',
      'salary',
      'credit',
    ];
    for (final kw in creditKeywords) {
      if (lower.contains(kw)) return TransactionType.credit;
    }

    // Debit signals
    const debitKeywords = [
      'debited',
      'deducted',
      'spent',
      'paid',
      'payment',
      'withdrawn',
      'withdrawal',
      'purchase',
      'debit',
    ];
    for (final kw in debitKeywords) {
      if (lower.contains(kw)) return TransactionType.debit;
    }

    // Default to debit — most financial alerts are expense notifications
    return TransactionType.debit;
  }

  // ─── Merchant Extraction ──────────────────────────────────────────────────

  /// Extracts merchant name from SMS body.
  /// Tries multiple patterns in priority order.
  String extractMerchant(String body) {
    // Pattern 1: "at <Merchant>" — most common in card/UPI SMS
    // e.g. "spent at Swiggy", "paid at Amazon"
    final atPattern = RegExp(
      r'(?:at|to|towards)\s+([A-Za-z0-9&\-\'\. ]{2,30}?)(?:\s+on|\s+for|\s+via|\s*\.|\s*,|\s*Rs|\s*INR|\s*₹|$)',
      caseSensitive: false,
    );
    final atMatch = atPattern.firstMatch(body);
    if (atMatch != null) {
      final candidate = _cleanMerchant(atMatch.group(1) ?? '');
      if (candidate.isNotEmpty) return candidate;
    }

    // Pattern 2: "VPA <merchant>@<bank>" — UPI transactions
    // e.g. "UPI to swiggy@icici"
    final vpaPattern = RegExp(
      r'(?:VPA|UPI ID|vpa)\s+([a-zA-Z0-9\.\-]+)@',
      caseSensitive: false,
    );
    final vpaMatch = vpaPattern.firstMatch(body);
    if (vpaMatch != null) {
      final candidate = _cleanMerchant(vpaMatch.group(1) ?? '');
      if (candidate.isNotEmpty) return candidate;
    }

    // Pattern 3: "Info: <Merchant>" — some bank formats
    final infoPattern = RegExp(
      r'(?:Info|Ref|Remarks?|Description)\s*[:\-]\s*([A-Za-z0-9 &\-\.]{2,30})',
      caseSensitive: false,
    );
    final infoMatch = infoPattern.firstMatch(body);
    if (infoMatch != null) {
      final candidate = _cleanMerchant(infoMatch.group(1) ?? '');
      if (candidate.isNotEmpty) return candidate;
    }

    // Pattern 4: Well-known merchant name anywhere in the SMS body
    final known = _findKnownMerchant(body);
    if (known != null) return known;

    // Fallback
    return 'Unknown';
  }

  // ─── Bank Extraction ──────────────────────────────────────────────────────

  /// Extracts the bank name from SMS sender or body.
  String extractBank(String body, String sender) {
    // Try sender short code first (most reliable)
    final senderBank = _bankFromSender(sender.toUpperCase());
    if (senderBank != null) return senderBank;

    // Fallback: look for bank name in SMS body
    final lower = body.toLowerCase();
    for (final entry in _bankKeywords.entries) {
      if (lower.contains(entry.key)) return entry.value;
    }

    return 'Unknown Bank';
  }

  // ─── Date Extraction ──────────────────────────────────────────────────────

  /// Extracts date from SMS body. Returns null if not found (caller uses SMS timestamp).
  DateTime? extractDate(String body) {
    // Pattern 1: DD-MM-YYYY or DD/MM/YYYY
    final dmyPattern = RegExp(
      r'(\d{2})[\/\-](\d{2})[\/\-](\d{4})',
    );
    final dmyMatch = dmyPattern.firstMatch(body);
    if (dmyMatch != null) {
      final day = int.tryParse(dmyMatch.group(1)!);
      final month = int.tryParse(dmyMatch.group(2)!);
      final year = int.tryParse(dmyMatch.group(3)!);
      if (day != null && month != null && year != null) {
        if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
          return DateTime(year, month, day);
        }
      }
    }

    // Pattern 2: DD MMM YYYY or DD-MMM-YYYY
    // e.g. "14 Jun 2025" or "14-Jun-2025"
    final dmonthPattern = RegExp(
      r'(\d{1,2})[\s\-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-,]*(\d{4})',
      caseSensitive: false,
    );
    final dmonthMatch = dmonthPattern.firstMatch(body);
    if (dmonthMatch != null) {
      final day = int.tryParse(dmonthMatch.group(1)!);
      final month = _monthFromAbbr(dmonthMatch.group(2)!);
      final year = int.tryParse(dmonthMatch.group(3)!);
      if (day != null && month != null && year != null) {
        return DateTime(year, month, day);
      }
    }

    // Pattern 3: YYYY-MM-DD (ISO format)
    final isoPattern = RegExp(r'(\d{4})-(\d{2})-(\d{2})');
    final isoMatch = isoPattern.firstMatch(body);
    if (isoMatch != null) {
      final year = int.tryParse(isoMatch.group(1)!);
      final month = int.tryParse(isoMatch.group(2)!);
      final day = int.tryParse(isoMatch.group(3)!);
      if (year != null && month != null && day != null) {
        return DateTime(year, month, day);
      }
    }

    return null;
  }

  // ─── Hash Generation ──────────────────────────────────────────────────────

  /// SHA-256 of (SMS body + timestamp) — used for duplicate detection.
  String _generateHash(String body, int timestampMs) {
    final input = '$body|$timestampMs';
    final bytes = utf8.encode(input);
    return sha256.convert(bytes).toString();
  }

  // ─── Private Helpers ──────────────────────────────────────────────────────

  String _cleanMerchant(String raw) {
    return raw
        .trim()
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'[^A-Za-z0-9 &\-\'\.]'), '')
        .trim();
  }

  /// Scans body for any well-known merchant name as a substring.
  String? _findKnownMerchant(String body) {
    final lower = body.toLowerCase();
    for (final name in _knownMerchants) {
      if (lower.contains(name)) {
        return name[0].toUpperCase() + name.substring(1);
      }
    }
    return null;
  }

  String? _bankFromSender(String sender) {
    for (final entry in _senderBankMap.entries) {
      if (sender.contains(entry.key)) return entry.value;
    }
    return null;
  }

  int? _monthFromAbbr(String abbr) {
    const months = {
      'jan': 1,
      'feb': 2,
      'mar': 3,
      'apr': 4,
      'may': 5,
      'jun': 6,
      'jul': 7,
      'aug': 8,
      'sep': 9,
      'oct': 10,
      'nov': 11,
      'dec': 12,
    };
    return months[abbr.toLowerCase().substring(0, 3)];
  }

  // ─── Static Lookup Tables ─────────────────────────────────────────────────

  static const List<String> _knownMerchants = [
    'swiggy', 'zomato', 'dominos', 'mcdonalds', 'kfc', 'subway',
    'starbucks', 'dunkin',
    'blinkit', 'zepto', 'bigbasket', 'dmart', 'jiomart',
    'uber', 'ola', 'rapido', 'irctc', 'makemytrip', 'redbus',
    'amazon', 'flipkart', 'myntra', 'ajio', 'meesho', 'nykaa',
    'airtel', 'jio', 'vodafone', 'bescom',
    'netflix', 'spotify', 'bookmyshow', 'hotstar', 'disney',
    'apollo', 'pharmaeasy', 'netmeds', '1mg',
    'phonepe', 'paytm', 'gpay', 'googlepay',
  ];

  static const Map<String, String> _senderBankMap = {
    'HDFCBK': 'HDFC Bank',
    'HDFC': 'HDFC Bank',
    'ICICIB': 'ICICI Bank',
    'ICICI': 'ICICI Bank',
    'SBIINB': 'SBI',
    'SBISMS': 'SBI',
    'SBI': 'SBI',
    'AXISBK': 'Axis Bank',
    'AXIS': 'Axis Bank',
    'KOTAKB': 'Kotak Bank',
    'KOTAK': 'Kotak Bank',
    'YESBNK': 'Yes Bank',
    'INDUSB': 'IndusInd Bank',
    'BOIIND': 'Bank of India',
    'CANBNK': 'Canara Bank',
    'PNBSMS': 'Punjab National Bank',
    'CENTBK': 'Central Bank',
    'IDBIBK': 'IDBI Bank',
    'FEDBK': 'Federal Bank',
    'PAYTMB': 'Paytm Bank',
    'SCBANK': 'Standard Chartered',
    'IOBSMS': 'Indian Overseas Bank',
  };

  static const Map<String, String> _bankKeywords = {
    'hdfc': 'HDFC Bank',
    'icici': 'ICICI Bank',
    'sbi': 'SBI',
    'axis bank': 'Axis Bank',
    'kotak': 'Kotak Bank',
    'yes bank': 'Yes Bank',
    'indusind': 'IndusInd Bank',
    'bank of india': 'Bank of India',
    'canara': 'Canara Bank',
    'punjab national': 'Punjab National Bank',
    'central bank': 'Central Bank',
    'idbi': 'IDBI Bank',
    'federal bank': 'Federal Bank',
    'paytm': 'Paytm Bank',
    'standard chartered': 'Standard Chartered',
  };
}
