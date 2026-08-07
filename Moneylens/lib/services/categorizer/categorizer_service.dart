import 'package:moneylens/core/constants/category_constants.dart';
import 'package:moneylens/models/transaction_model.dart';
import 'package:moneylens/repositories/merchant_repository.dart';

/// Result of a single categorization attempt — useful for debugging + tests
class CategorizationResult {
  final String merchant;
  final String category;
  final CategorizationSource source;

  const CategorizationResult({
    required this.merchant,
    required this.category,
    required this.source,
  });

  @override
  String toString() =>
      'CategorizationResult(merchant: $merchant, category: $category, via: ${source.name})';
}

/// How the category was resolved — for logging and future ML training data
enum CategorizationSource {
  exactDb,
  partialDb,
  fuzzyKeyword,
  transactionType,
  fallback,
}

// ─────────────────────────────────────────────────────────────────────────────

class CategorizerService {
  CategorizerService._();
  static final CategorizerService instance = CategorizerService._();

  final _merchantRepo = MerchantRepository.instance;

  // ─── Public API ───────────────────────────────────────────────────────────

  /// Categorize a single merchant name → category string.
  Future<String> categorize(
    String merchant, {
    TransactionType type = TransactionType.debit,
    String? smsBody,
  }) async {
    final result = await categorizeWithSource(
      merchant,
      type: type,
      smsBody: smsBody,
    );
    return result.category;
  }

  /// Categorize and return full result including how the match was found.
  Future<CategorizationResult> categorizeWithSource(
    String merchant, {
    TransactionType type = TransactionType.debit,
    String? smsBody,
  }) async {
    final cleaned = _cleanMerchant(merchant);

    // ── Step 1: Exact match in DB ────────────────────────────────────────────
    final dbCategory = await _exactDbLookup(cleaned);
    if (dbCategory != null) {
      return CategorizationResult(
        merchant: cleaned,
        category: dbCategory,
        source: CategorizationSource.exactDb,
      );
    }

    // ── Step 2: Partial match in DB ──────────────────────────────────────────
    final partialCategory = await _partialDbLookup(cleaned);
    if (partialCategory != null) {
      return CategorizationResult(
        merchant: cleaned,
        category: partialCategory,
        source: CategorizationSource.partialDb,
      );
    }

    // ── Step 3: Fuzzy keyword match against SMS body ──────────────────────────
    if (smsBody != null && smsBody.isNotEmpty) {
      final bodyCategory = _fuzzyBodyMatch(smsBody);
      if (bodyCategory != null) {
        return CategorizationResult(
          merchant: cleaned,
          category: bodyCategory,
          source: CategorizationSource.fuzzyKeyword,
        );
      }
    }

    // ── Step 4: Fuzzy keyword match against merchant name itself ─────────────
    final merchantCategory = _fuzzyBodyMatch(cleaned);
    if (merchantCategory != null) {
      return CategorizationResult(
        merchant: cleaned,
        category: merchantCategory,
        source: CategorizationSource.fuzzyKeyword,
      );
    }

    // ── Step 5: Derive from transaction type ─────────────────────────────────
    if (type == TransactionType.credit) {
      final creditCategory = _categoryFromCreditType(cleaned, smsBody ?? '');
      if (creditCategory != null) {
        return CategorizationResult(
          merchant: cleaned,
          category: creditCategory,
          source: CategorizationSource.transactionType,
        );
      }
    }

    // ── Step 6: Fallback ──────────────────────────────────────────────────────
    return CategorizationResult(
      merchant: cleaned,
      category: CategoryConstants.others,
      source: CategorizationSource.fallback,
    );
  }

  /// Batch categorize — applies to a list of transactions in place.
  /// Returns a new list with category fields filled in.
  Future<List<TransactionModel>> categorizeAll(
    List<TransactionModel> transactions,
  ) async {
    // Pre-load merchant cache once before looping
    await _merchantRepo.getAllMappings();

    final result = <TransactionModel>[];
    for (final tx in transactions) {
      final category = await categorize(
        tx.merchant,
        type: tx.transactionType,
        smsBody: tx.smsBody,
      );
      result.add(tx.copyWith(category: category));
    }
    return result;
  }

  /// Update a merchant's category in DB and invalidate cache.
  /// Called when the user manually re-categorizes a transaction.
  Future<void> updateMerchantCategory(
    String merchant,
    String newCategory,
  ) async {
    await _merchantRepo.upsertMerchant(
      _cleanMerchant(merchant),
      newCategory,
    );
  }

  /// Reload the merchant mapping cache (e.g. after bulk update).
  Future<void> reloadMappings() async {
    await _merchantRepo.invalidateCache();
  }

  // ─── Step Implementations ─────────────────────────────────────────────────

  Future<String?> _exactDbLookup(String merchant) async {
    final mappings = await _merchantRepo.getAllMappings();
    final key = merchant.toLowerCase();
    return mappings[key];
  }

  Future<String?> _partialDbLookup(String merchant) async {
    final mappings = await _merchantRepo.getAllMappings();
    final key = merchant.toLowerCase();

    // Check if any DB key is contained in the merchant name or vice versa
    for (final entry in mappings.entries) {
      if (key.contains(entry.key) || entry.key.contains(key)) {
        return entry.value;
      }
    }
    return null;
  }

  /// Scans the text for category-specific keywords.
  String? _fuzzyBodyMatch(String text) {
    final lower = text.toLowerCase();
    for (final entry in _categoryKeywords.entries) {
      for (final keyword in entry.value) {
        if (lower.contains(keyword)) return entry.key;
      }
    }
    return null;
  }

  /// For credit transactions, try to guess salary vs refund vs transfer.
  String? _categoryFromCreditType(String merchant, String body) {
    final lower = body.toLowerCase();
    if (lower.contains('salary') ||
        lower.contains('payroll') ||
        lower.contains('stipend')) {
      return CategoryConstants.salary;
    }
    if (lower.contains('refund') || lower.contains('reversal')) {
      return CategoryConstants.shopping;
    }
    if (lower.contains('neft') ||
        lower.contains('imps') ||
        lower.contains('transfer')) {
      return CategoryConstants.transfer;
    }
    return null;
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────

  String _cleanMerchant(String raw) {
    return raw
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'[^a-z0-9 &\-\'\.]'), '');
  }

  // ─── Category Keyword Map ─────────────────────────────────────────────────
  // Used for fuzzy matching when DB lookup fails.
  // Keywords are lowercase substrings to match against merchant name or body.

  static const Map<String, List<String>> _categoryKeywords = {
    CategoryConstants.food: [
      'swiggy', 'zomato', 'domino', 'pizza', 'burger', 'kfc', 'mcdonald',
      'subway', 'starbucks', 'dunkin', 'restaurant', 'cafe', 'food',
      'biryani', 'hotel', 'dine', 'eat', 'kitchen', 'barbeque', 'bbq',
    ],

    CategoryConstants.grocery: [
      'blinkit', 'zepto', 'bigbasket', 'big basket', 'dmart', 'jiomart',
      'grofers', 'grocery', 'supermarket', 'hypermarket', 'nature basket',
      'more retail', 'reliance fresh', 'spencer',
    ],

    CategoryConstants.travel: [
      'uber', 'ola', 'rapido', 'namma metro', 'bmtc', 'ksrtc', 'irctc',
      'makemytrip', 'goibibo', 'redbus', 'yatra', 'cleartrip', 'airline',
      'airways', 'indigo', 'spicejet', 'vistara', 'air india', 'metro',
      'cab', 'taxi', 'auto', 'train', 'flight', 'bus',
    ],

    CategoryConstants.shopping: [
      'amazon', 'flipkart', 'myntra', 'ajio', 'meesho', 'nykaa', 'snapdeal',
      'tatacliq', 'shopsy', 'reliance digital', 'croma', 'vijay sales',
      'shopping', 'mall', 'retail', 'store', 'shop',
    ],

    CategoryConstants.bills: [
      'airtel', 'jio', 'vodafone', 'vi ', 'bsnl', 'bescom', 'bwssb', 'bbmp',
      'tata power', 'adani electricity', 'mseb', 'electricity', 'water bill',
      'gas', 'broadband', 'recharge', 'postpaid', 'prepaid', 'bill payment',
      'utility',
    ],

    CategoryConstants.entertainment: [
      'netflix', 'spotify', 'bookmyshow', 'hotstar', 'disney', 'prime video',
      'youtube', 'zee5', 'sonyliv', 'mxplayer', 'jiosaavn', 'gaana',
      'pvr', 'inox', 'multiplex', 'cinema', 'movie', 'concert', 'event',
    ],

    CategoryConstants.medical: [
      'apollo', 'pharmaeasy', 'netmeds', '1mg', 'practo', 'medplus',
      'hospital', 'clinic', 'pharmacy', 'medical', 'health', 'doctor',
      'diagnostic', 'lab', 'pathology', 'dentist', 'dental', 'optician',
    ],

    CategoryConstants.fuel: [
      'hpcl', 'bpcl', 'iocl', 'shell', 'petrol', 'diesel', 'fuel',
      'hp pump', 'indian oil', 'bharat petroleum', 'hindustan petroleum',
      'cng', 'ev charge', 'charging station',
    ],

    CategoryConstants.salary: [
      'salary', 'payroll', 'stipend', 'wages', 'remuneration',
    ],

    CategoryConstants.transfer: [
      'neft', 'imps', 'rtgs', 'transfer', 'upi', 'sent to', 'transferred',
    ],
  };
}
